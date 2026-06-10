// Сервис эмбеддингов для семантического поиска. Считает векторы текста через
// локальную модель (Transformers.js, multilingual-e5-small) — без API-ключей.
// Модель ~110 МБ скачивается один раз и кешируется на диске. Векторы хранятся
// в Postgres (Episode.embedding, pgvector); ранжирование — SQL-оператором <=>
// (см. routes/search.ts), индексация — в cron-refresh (lib/embedEpisodes.ts).
//
// Дизайн повторяет паттерн «optional integration» из lib/adapty.ts: пока
// модель не загружена/недоступна — ручка отдаёт 503, остальной бэкенд работает
// как раньше. Текстовый поиск в iOS-клиенте продолжает работать независимо:
// семантика — дополнительный слой, а не замена.
//
// Префиксы "query:" / "passage:" обязательны для e5-моделей — без них
// качество заметно падает (так предписывает карточка модели intfloat/e5).

import type { FeatureExtractionPipeline } from "@xenova/transformers";

const MODEL_ID = "Xenova/multilingual-e5-small";

let pipelinePromise: Promise<FeatureExtractionPipeline> | null = null;

async function getPipeline(): Promise<FeatureExtractionPipeline> {
  if (!pipelinePromise) {
    // Динамический импорт: тяжёлую зависимость не тянем в память, пока
    // семантический поиск реально не понадобился. Если пакет не установлен
    // или модель не скачалась — отлетит сюда, ручка превратит это в 503.
    pipelinePromise = import("@xenova/transformers").then(({ pipeline }) =>
      pipeline("feature-extraction", MODEL_ID),
    );
  }
  return pipelinePromise;
}

async function embed(text: string, kind: "query" | "passage"): Promise<number[]> {
  const pipe = await getPipeline();
  const output = await pipe(`${kind}: ${text}`, { pooling: "mean", normalize: true });
  return Array.from(output.data as Float32Array);
}

// Текст для эмбеддинга эпизода. title несёт основную смысловую нагрузку,
// summary уточняет. Обрезаем summary, чтобы не упереться в лимит модели
// (512 токенов) и не размывать вектор слишком длинным описанием.
//
// ВАЖНО: SQL-выражение в lib/embedEpisodes.ts (детект устаревших векторов)
// должно точно повторять эту функцию. Меняешь её — меняй и SQL, иначе cron
// будет пересчитывать весь корпус на каждом прогоне.
export function episodeEmbeddingText(ep: { title: string; summary: string }): string {
  const summary = ep.summary.slice(0, 1000);
  return summary ? `${ep.title}. ${summary}` : ep.title;
}

// Эмбеддинг эпизода (passage) — для cron при индексации.
export function embedPassage(text: string): Promise<number[]> {
  return embed(text, "passage");
}

// Эмбеддинг запроса (query) — для ручки поиска.
export function embedQuery(text: string): Promise<number[]> {
  return embed(text, "query");
}

// Прогрев модели в фоне при старте сервера (см. server.ts): без него первый
// поисковый запрос после деплоя ждал бы скачивания и инициализации модели.
export function warmupModel(): Promise<unknown> {
  return getPipeline();
}
