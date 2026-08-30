# LFU-Light - A Lightweight Least Frequently Used Cache

`lfu-light` is a lightweight LFU-Cache implementation with a lightweight API which operates in O(1) time with minimal footprint.

This project was inspired by [LRU](https://github.com/jeromefroe/lru-rs)


## Example

```rust
use lfu_light::LfuCache;

fn main() {
    let mut lfu = LfuCache::with_capacity(2);
    lfu.put("apple", 1);
    lfu.put("banana", 2);
    assert_eq!(lfu.get("apple"), Some(&1));

    // Banana is least frequently used so is evicted
    lfu.put("orange", 3);
    assert!(lfu.get("banana").is_none());

    assert_eq!(lfu.put("orange", 5), Some(3));
    *lfu.get_mut("apple").expect("apple is still present") = 12;
    assert_eq!(lfu.get("apple"), Some(&12));

    assert_eq!(lfu.remove("orange"), Some(5));
}
```
