ngt

(base) adel@192 munind % time vendors/ngt/build/bin/ngt/ngt \
  create \
  -d 100 \
  -D c \
  benches/indexes/glove-100-angular-ngt \
  benches/data/glove-100-angular.train.tsv
vendors/ngt/build/bin/ngt/ngt create -d 100 -D c    477.23s user 7.28s system 442% cpu 1:49.51 total

(base) adel@192 munind % python3 scripts/eval_ngt_recall.py
Loaded ground truth: 10000 queries, top-10

-e 0.1
  recall@10: 0.628270
  avg_query_ms: 0.272487

-e 0.4
  recall@10: 0.978710
  avg_query_ms: 15.525700


munind

(base) adel@192 munind % time target/release/munind create \  
  -d 100 \
  -D c \
  benches/indexes/glove-100-angular-munind \
  benches/data/glove-100-angular.train.tsv
Reading benches/data/glove-100-angular.train.tsv... 1183514 vectors in 4.40s
Inserting... done in 0.24s
Building index... done in 109.73s
Saving index... done in 0.38s
munind: created index at benches/indexes/glove-100-angular-munind with 1183514 objects
target/release/munind create -d 100 -D c    697.24s user 37.25s system 636% cpu 1:55.34 total
(base) adel@192 munind % python3 scripts/eval_munind_recall.py
Loaded ground truth: 10000 queries, top-10

-e 0.1
  recall@10: 0.635250
  avg_query_ms: 0.257562

-e 0.4
  recall@10: 0.986620
  avg_query_ms: 16.603397

