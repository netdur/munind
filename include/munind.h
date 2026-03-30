/*
 * munind — Fast approximate nearest neighbor search library.
 *
 * C API header.
 *
 * Build:  cargo build --release
 * Link:   -lmunind (from target/release/)
 */

#ifndef MUNIND_H
#define MUNIND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque index handle. */
typedef struct MunindIndex MunindIndex;

/* Search result. */
typedef struct {
    uint32_t id;
    float distance;
} MunindResult;

/*
 * Create a new empty index.
 * distance: "l2", "cosine", "ip", "l1", "angle" (NULL defaults to "l2").
 * Returns NULL on error.
 */
MunindIndex* munind_create(uint32_t dimension, const char* distance);

/*
 * Open an existing index from a directory.
 * Returns NULL on error.
 */
MunindIndex* munind_open(const char* path);

/* Free an index. Safe to call with NULL. */
void munind_free(MunindIndex* index);

/*
 * Insert a vector. Returns the assigned ID (1-based), or 0 on error.
 * dim must match the index dimension.
 */
uint32_t munind_insert(MunindIndex* index, const float* vector, uint32_t dim);

/* Build the graph index. Call after all inserts. */
void munind_build(MunindIndex* index);

/* Save the index to a directory. Returns 0 on success, -1 on error. */
int munind_save(const MunindIndex* index, const char* path);

/*
 * Search for k nearest neighbors.
 *
 * results:      caller-allocated array of at least k MunindResult.
 * result_count: set to actual number of results found (<= k).
 * epsilon:      exploration coefficient (0.1 = default, higher = slower + better recall).
 *
 * Returns 0 on success, -1 on error.
 */
int munind_search(const MunindIndex* index,
                  const float* query, uint32_t dim,
                  uint32_t k, float epsilon,
                  MunindResult* results, uint32_t* result_count);

/* Return the number of objects in the index. */
uint32_t munind_len(const MunindIndex* index);

/* Return the vector dimension. */
uint32_t munind_dimension(const MunindIndex* index);

#ifdef __cplusplus
}
#endif

#endif /* MUNIND_H */
