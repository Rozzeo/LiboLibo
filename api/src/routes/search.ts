import { Router } from "express";
import pgvector from "pgvector";
import { prisma } from "../db.js";
import { asyncHandler } from "../lib/asyncHandler.js";
import { episodeToDTO } from "../lib/serialize.js";
import { resolveViewer } from "../middleware/viewer.js";
import { embedQuery } from "../lib/semanticSearch.js";

export const searchRouter = Router();

const MAX_QUERY_LEN = 200;
const DEFAULT_LIMIT = 20;
const MAX_LIMIT = 50;

// Порог косинусной БЛИЗОСТИ (1 - distance). e5 для русского даёт ~0.78+ на
// релевантных парах, поэтому 0.75 отсекает явный шум. Вынесено в env для
// тюнинга без передеплоя; guard на NaN — мусор в env откатывается на дефолт.
function readMinScore(): number {
  const n = Number(process.env.SEMANTIC_MIN_SCORE ?? "0.75");
  return Number.isFinite(n) ? n : 0.75;
}

interface SemanticBody {
  query?: unknown;
  limit?: unknown;
}

interface HitRow {
  id: string;
  score: number;
}

// POST /v1/search/semantic — семантический поиск по выпускам всех подкастов.
// Дополняет (не заменяет) текстовый поиск iOS-клиента: находит выпуски по
// смыслу запроса, даже если точных слов нет в title/summary.
//
// Тело: { "query": "как пережить расставание", "limit": 20 }
// Ответ: { "items": EpisodeDTO[] } — тот же формат, что /v1/feed, чтобы
// клиент переиспользовал существующую модель Episode и рендер ячеек.
//
// Эмбеддинги эпизодов лежат в Postgres (pgvector) и считаются cron-refresh'ем
// заранее — здесь только эмбеддинг запроса и один SQL-запрос с <=>.
searchRouter.post(
  "/search/semantic",
  resolveViewer,
  asyncHandler(async (req, res) => {
    const body = (req.body ?? {}) as SemanticBody;

    const query = typeof body.query === "string" ? body.query.trim() : "";
    if (query.length === 0) {
      return res.status(400).json({ error: "missing_query" });
    }
    if (query.length > MAX_QUERY_LEN) {
      return res.status(400).json({ error: "query_too_long" });
    }

    const limit = clampLimit(body.limit);
    const minScore = readMinScore();

    // Эмбеддинг запроса. Если модель недоступна (пакет не установлен, не
    // скачалась) — 503, клиент откатывается на текстовый поиск.
    let queryVec: number[];
    try {
      queryVec = await embedQuery(query);
    } catch (err) {
      console.error("[search/semantic] model unavailable:", err);
      return res.status(503).json({ error: "semantic_unavailable" });
    }

    // pgvector: <=> — косинусная ДИСТАНЦИЯ (0 = идентичны, 2 = противоположны),
    // score = 1 - distance. Фильтруем по minScore, сортируем по дистанции,
    // режем по limit. Один запрос, без скана корпуса в память. Premium-эпизоды
    // ранжируются наравне — audio_url гейтится в episodeToDTO как везде.
    const vecSql = pgvector.toSql(queryVec);
    const rows = await prisma.$queryRaw<HitRow[]>`
      SELECT id, 1 - (embedding <=> ${vecSql}::vector) AS score
      FROM episodes
      WHERE embedding IS NOT NULL
        AND 1 - (embedding <=> ${vecSql}::vector) >= ${minScore}
      ORDER BY embedding <=> ${vecSql}::vector
      LIMIT ${limit}
    `;

    if (rows.length === 0) {
      return res.json({ items: [] });
    }

    // Дотягиваем полные эпизоды + podcast, сохраняя порядок ранжирования
    // (Prisma вернёт строки в произвольном порядке).
    const ids = rows.map((r) => r.id);
    const episodes = await prisma.episode.findMany({
      where: { id: { in: ids } },
      include: { podcast: { select: { name: true, artworkUrl: true } } },
    });
    const byId = new Map(episodes.map((e) => [e.id, e]));

    const items = rows
      .map((r) => byId.get(r.id))
      .filter((e): e is NonNullable<typeof e> => e != null)
      .map((e) => episodeToDTO(e, e.podcast, req.viewer));

    res.json({ items });
  }),
);

// GET /v1/search/semantic/status — диагностика: сколько эпизодов
// проиндексировано. Удобно для health-проверки фичи на проде.
searchRouter.get(
  "/search/semantic/status",
  asyncHandler(async (_req, res) => {
    const [{ count }] = await prisma.$queryRaw<{ count: bigint }[]>`
      SELECT COUNT(*)::bigint AS count FROM episodes WHERE embedding IS NOT NULL
    `;
    res.json({ indexed: Number(count) });
  }),
);

function clampLimit(raw: unknown): number {
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) return DEFAULT_LIMIT;
  return Math.min(Math.floor(n), MAX_LIMIT);
}
