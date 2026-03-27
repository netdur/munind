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
munind: created index at benches/indexes/glove-100-angular-munind with 1183514 objects
target/release/munind create -d 100 -D c    225.43s user 27.48s system 434% cpu 58.237 total
(base) adel@192 munind % python3 scripts/eval_munind_recall.py
Loaded ground truth: 10000 queries, top-10

-e 0.1
  recall@10: 0.622100
  avg_query_ms: 0.211457

-e 0.4
  recall@10: 0.981760
  avg_query_ms: 12.146029

