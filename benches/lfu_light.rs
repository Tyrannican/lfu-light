use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Zipf};
use std::{collections::HashSet, hint::black_box};

use lfu_light::LfuCache;

const CAPACITY: usize = 10_000;
const ACCESSES: u64 = 100_000;

fn dist(count: usize, total_accesses: u64, skew: f64, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Zipf::new(total_accesses as f64, skew).expect("total_accesses >= 0 and skew >= 0");
    (0..count)
        .map(|_| dist.sample(&mut rng).max(1.0) as u64 - 1)
        .collect()
}

fn setup_cache(capacity: usize, total_accesses: u64, seed: u64) -> LfuCache<u64, u64> {
    let mut cache = LfuCache::with_capacity(capacity);
    for k in dist(capacity.saturating_mul(20), total_accesses, 0.99, seed) {
        if cache.get(&k).is_none() {
            cache.put(k, k);
        }
    }

    cache
}

fn hot_paths(
    cache: &LfuCache<u64, u64>,
    total_accesses: u64,
    target: usize,
    seed: u64,
) -> Vec<u64> {
    let mut s = HashSet::new();
    let keys: Vec<u64> = dist(target * 8, total_accesses, 1.2, seed)
        .into_iter()
        .filter(|k| cache.peek(k).is_some() && s.insert(*k))
        .take(target)
        .collect();

    assert!(!keys.is_empty());
    keys
}

fn bench_operations(c: &mut Criterion) {
    let mask = 1023;
    let cache = setup_cache(CAPACITY, ACCESSES, 0xdeadbeef);
    let hot_paths = hot_paths(&cache, ACCESSES, 1024, 0xdeedbeed);
    let mut group = c.benchmark_group("operations");
    group.throughput(Throughput::Elements(1));

    group.bench_function("peek", |b| {
        let mut i = 0;
        b.iter(|| {
            i = (i + 1) & mask;
            black_box(cache.peek(black_box(&hot_paths[i])))
        })
    });

    let mut tmp_cache = cache.clone();
    group.bench_function("get", |b| {
        let mut i = 0;
        b.iter(|| {
            i = (i + 1) & mask;
            black_box(tmp_cache.get(black_box(&hot_paths[i])));
        })
    });

    let mut tmp_cache = cache.clone();
    group.bench_function("get_cachemiss", |b| {
        let mut i = ACCESSES;
        b.iter(|| {
            i += 1;
            black_box(tmp_cache.get(black_box(&i)));
        })
    });

    group.bench_function("contain_miss", |b| {
        let mut i = ACCESSES;
        b.iter(|| {
            i += 1;
            black_box(cache.contains_key(black_box(&i)));
        })
    });

    let mut update_cache = cache.clone();
    group.bench_function("put_update", |b| {
        let mut i = 0;
        b.iter(|| {
            i = (i + 1) & mask;
            black_box(update_cache.put(black_box(hot_paths[i]), black_box(7)));
        })
    });

    let mut update_cache = cache.clone();
    group.bench_function("put_eviction", |b| {
        let mut idx = ACCESSES;
        b.iter(|| {
            idx += 1;
            black_box(update_cache.put(black_box(idx), black_box(0)));
        })
    });

    let mut insert_cache = cache.clone();
    group.bench_function("get_or_insert_hit", |b| {
        let mut i = 0;
        b.iter(|| {
            i = (i + 1) & mask;
            black_box(*insert_cache.get_or_insert(black_box(hot_paths[i]), 0))
        })
    });

    let mut insert_cache = cache.clone();
    group.bench_function("manual_get_or_insert", |b| {
        let mut i = 0;
        b.iter(|| {
            i = (i + 1) & mask;
            let k = black_box(hot_paths[i]);
            if insert_cache.get(&k).is_none() {
                insert_cache.put(k, 0);
            }
        })
    });

    group.throughput(Throughput::Elements(2));

    let mut remove_cache = cache.clone();
    group.bench_function("remove_and_insert", |b| {
        let mut i = 0;
        b.iter(|| {
            i = (i + 1) & mask;
            let k = hot_paths[i];
            let v = black_box(remove_cache.remove(black_box(&k))).expect("value is present");
            black_box(remove_cache.put(k, v));
        })
    });

    let mut evict_cache = cache.clone();
    group.bench_function("eviction", |b| {
        b.iter(|| {
            let (k, v) = black_box(evict_cache.evict()).expect("not none");
            black_box(evict_cache.put(k, v));
        })
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(3))
        .measurement_time(std::time::Duration::from_secs(10))
        .sample_size(200)
        .noise_threshold(0.03)
        .significance_level(0.01);
    targets = bench_operations
}
criterion_main!(benches);
