import { describe, it, expect } from "vitest";
import { cosineSimilarity, rankBySimilarity } from "../src/lib/semanticRank.js";

describe("cosineSimilarity", () => {
  it("returns 1 for identical vectors", () => {
    expect(cosineSimilarity([1, 2, 3], [1, 2, 3])).toBeCloseTo(1, 10);
  });

  it("returns 1 for parallel vectors of different magnitude", () => {
    expect(cosineSimilarity([1, 0], [5, 0])).toBeCloseTo(1, 10);
  });

  it("returns 0 for orthogonal vectors", () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0, 10);
  });

  it("returns -1 for opposite vectors", () => {
    expect(cosineSimilarity([1, 1], [-1, -1])).toBeCloseTo(-1, 10);
  });

  it("returns 0 when either vector is all zeros (no NaN)", () => {
    expect(cosineSimilarity([0, 0], [1, 1])).toBe(0);
    expect(cosineSimilarity([1, 1], [0, 0])).toBe(0);
  });

  it("throws on length mismatch", () => {
    expect(() => cosineSimilarity([1, 2], [1, 2, 3])).toThrow(/length mismatch/);
  });
});

describe("rankBySimilarity", () => {
  // Query points roughly toward the first candidate.
  const query = [1, 0];
  const candidates = [
    { item: "a", vector: [1, 0] }, // identical → score 1
    { item: "b", vector: [0.7, 0.7] }, // 45° → ~0.707
    { item: "c", vector: [0, 1] }, // orthogonal → 0
    { item: "d", vector: [-1, 0] }, // opposite → -1
  ];

  it("orders by descending similarity, dropping negatives by default", () => {
    // Default minScore is 0, so "d" (opposite, score -1) is filtered out.
    const ranked = rankBySimilarity(query, candidates, { limit: 10 });
    expect(ranked.map((r) => r.item)).toEqual(["a", "b", "c"]);
  });

  it("includes negatives when minScore is lowered below them", () => {
    const ranked = rankBySimilarity(query, candidates, { limit: 10, minScore: -2 });
    expect(ranked.map((r) => r.item)).toEqual(["a", "b", "c", "d"]);
  });

  it("respects the limit", () => {
    const ranked = rankBySimilarity(query, candidates, { limit: 2 });
    expect(ranked.map((r) => r.item)).toEqual(["a", "b"]);
  });

  it("filters out candidates below minScore", () => {
    const ranked = rankBySimilarity(query, candidates, { limit: 10, minScore: 0.5 });
    expect(ranked.map((r) => r.item)).toEqual(["a", "b"]);
  });

  it("returns empty array when nothing passes minScore", () => {
    const ranked = rankBySimilarity(query, candidates, { limit: 10, minScore: 1.1 });
    expect(ranked).toEqual([]);
  });

  it("is stable: equal scores keep input order", () => {
    const tied = [
      { item: "x", vector: [1, 0] },
      { item: "y", vector: [1, 0] },
      { item: "z", vector: [1, 0] },
    ];
    const ranked = rankBySimilarity([1, 0], tied, { limit: 10 });
    expect(ranked.map((r) => r.item)).toEqual(["x", "y", "z"]);
  });

  it("attaches the computed score to each hit", () => {
    const ranked = rankBySimilarity(query, candidates, { limit: 1 });
    expect(ranked[0]!.item).toBe("a");
    expect(ranked[0]!.score).toBeCloseTo(1, 10);
  });
});
