//! Least Frequently Used Cache with a lightweight API.
//!
//! Entries that are the least frequently interacted with are evicted from the cache when it
//! reaches capacity or when the user manually evicts them.
//!
//! In the case where eviction candidates are tied, a Least Recently Used policy is applied to
//! select the appropriate candidate.
//!
//! # Aging
//!
//! An optional aging mechanism is also employed to prevent cache pollution in the case where certain
//! entries are extremenly hot and essentially become irrelevant.
//!
//! Any operation which interacts with an existing value (that isn't peeking or iterating) will
//! tick up the operations counter if an aging value is set.
//!
//! Once that threshold is met, ALL entries in the cache will have their frequency tracking value
//! halved.
//!
//! # Example
//!
//! ```rust
//! use lfu_light::LfuCache;
//!
//! fn main() {
//!     let mut lfu = LfuCache::with_capacity(2);
//!     lfu.put("apple", 1);
//!     lfu.put("banana", 2);
//!     assert_eq!(lfu.get("apple"), Some(&1));
//!
//!     // Banana is least frequently used so is evicted
//!     lfu.put("orange", 3);
//!     assert!(lfu.get("banana").is_none());
//!
//!     assert_eq!(lfu.put("orange", 5), Some(3));
//!     *lfu.get_mut("apple").expect("apple is still present") = 12;
//!     assert_eq!(lfu.get("apple"), Some(&12));
//!
//!     assert_eq!(lfu.remove("orange"), Some(5));
//! }
//! ```

#![deny(missing_docs)]
#![allow(clippy::manual_flatten)]

use core::{
    borrow::Borrow,
    hash::{BuildHasher, Hash},
    num::NonZeroUsize,
};
use hashbrown::{DefaultHashBuilder, HashTable};

type Index = u32;
const NULL_IDX: Index = Index::MAX;
const DEFAULT_CAPACITY: usize = 128;
const MAX_CAPACITY: usize = (NULL_IDX - 1) as usize;

/// Least Frequently Used Cache
///
/// Upon cache hitting capacity, new entries will evict the entry with the least
/// frequent access.
///
/// In the case an entry is tied for the least frequent accesses, a least recently used
/// policy is enacted.
pub struct LfuCache<K, V, S = DefaultHashBuilder> {
    len: usize,
    capacity: usize,
    age_interval: Option<NonZeroUsize>,
    op_counter: usize,

    key_map: HashTable<Index>,
    entries: Vec<Option<CacheEntry<K, V>>>,
    hasher: S,

    free_slots: Vec<Index>,
    freq_slots: Vec<FreqSlot>,
    free_freq_slot: Index,
    frequency_head: Index,
}

impl<K, V> LfuCache<K, V> {
    /// Constructs a new cache with default capacity and default hasher
    pub fn new() -> Self {
        Self::with_capacity_and_hasher(DEFAULT_CAPACITY, DefaultHashBuilder::default())
    }

    /// Constructs a new cache with `capacity` slots available and a default hasher
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, DefaultHashBuilder::default())
    }
}

impl<K, V, S> LfuCache<K, V, S> {
    /// Constructs a new cache with the given Hasher and default capacity
    pub fn with_hasher(hasher: S) -> Self {
        Self::with_capacity_and_hasher(DEFAULT_CAPACITY, hasher)
    }

    /// Constructs a new cache with the given Hasher and `capacity` slots available
    pub fn with_capacity_and_hasher(capacity: usize, hasher: S) -> Self {
        let capacity = capacity.clamp(1, MAX_CAPACITY / 2);
        Self {
            key_map: HashTable::with_capacity(capacity),
            entries: Vec::with_capacity(capacity),
            free_slots: Vec::new(),
            freq_slots: Vec::new(),
            frequency_head: NULL_IDX,
            free_freq_slot: NULL_IDX,
            hasher,
            len: 0,
            capacity,
            age_interval: None,
            op_counter: 0,
        }
    }

    /// An iterator which visits all key-value pairs (does not bump frequency count)
    ///
    /// Yields type `(&'a K, &'a V)`
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            slots: self.entries.iter(),
            remaining: self.len,
        }
    }

    /// A mutable iterator which visits all key-value pairs (does not bump frequency count)
    ///
    /// Yields type `(&'a mut K, &'a mut V)`
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            slots: self.entries.iter_mut(),
            remaining: self.len,
        }
    }

    /// An iterator which visits all keys in insertion order
    ///
    /// Yields type `&'a K`
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &K> {
        self.iter().map(|(k, _)| k)
    }

    /// An iterator which visits all values in insertion order
    ///
    /// Yields type `&'a V`
    pub fn values(&self) -> impl ExactSizeIterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }

    /// A mutable iterator which visits all values in insertion order
    ///
    /// Yields type `&'a mut V`
    pub fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut V> {
        self.iter_mut().map(|(_, v)| v)
    }
}

impl<K, V, S> LfuCache<K, V, S> {
    /// Clears all entries in the cache
    pub fn clear(&mut self) {
        self.entries.clear();
        self.key_map.clear();
        self.free_slots.clear();
        self.freq_slots.clear();
        self.free_freq_slot = NULL_IDX;
        self.frequency_head = NULL_IDX;
        self.len = 0;
        self.op_counter = 0;
    }

    /// Capacity of the cache
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current length of the cache
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Sets the interval after which the frequencies held within the cache are halved
    ///
    /// This is designed to prevent cache pollution where entries that are constantly accessed
    /// essentially become irrelevant to the cache.
    ///
    /// This will set the number of operations that are performed before all frequencies in the
    /// cache are halved
    pub fn age_after(mut self, interval: usize) -> Self {
        self.age_interval = NonZeroUsize::new(interval);
        self.op_counter = 0;
        self
    }
}

impl<K: Hash + Eq, V, S: BuildHasher> LfuCache<K, V, S> {
    /// Returns a reference to the value associated with the key
    #[inline]
    pub fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hasher.hash_one(key);
        let entry_idx = self.find(key, hash)?;
        self.bump(entry_idx);
        self.tick();

        let entry = self.fetch_entry(entry_idx);
        Some(&entry.value)
    }

    /// Returns a mutable reference to the value associated with the key
    #[inline]
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hasher.hash_one(key);
        let entry_idx = self.find(key, hash)?;
        self.bump(entry_idx);
        self.tick();

        let entry = self.fetch_entry_mut(entry_idx);
        Some(&mut entry.value)
    }

    /// Returns a reference to the value associated with the key or inserts it with the given value
    #[inline]
    pub fn get_or_insert(&mut self, key: K, insert_with: V) -> &V
    where
        K: Clone,
    {
        if self.get(&key).is_none() {
            self.put(key.clone(), insert_with);
        }

        self.peek(&key)
            .expect("key was either present or just inserted")
    }

    /// Returns a reference to the value associated with the key or inserts it with the value
    /// evaluated from the provided closure
    #[inline]
    pub fn get_or_insert_with<F>(&mut self, key: K, insert_with: F) -> &V
    where
        K: Clone,
        F: FnOnce() -> V,
    {
        self.get_or_insert(key, insert_with())
    }

    /// Returns `true` if the cache contains an entry for the given key
    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hasher.hash_one(key);
        self.find(key, hash).is_some()
    }

    /// Inserts a key-value pair into the cache. If this updates a currently existing value, the
    /// old value is returned
    #[inline]
    pub fn put(&mut self, key: K, value: V) -> Option<V> {
        let hash = self.hasher.hash_one(&key);
        if let Some(idx) = self.find(&key, hash) {
            let previous_entry = std::mem::replace(&mut self.fetch_entry_mut(idx).value, value);
            self.bump(idx);
            self.tick();
            return Some(previous_entry);
        }

        if self.is_full() {
            self.evict();
        }

        let entry = CacheEntry {
            key,
            value,
            hash,
            prev: NULL_IDX,
            next: NULL_IDX,
            slot_index: NULL_IDX,
        };

        let insert_idx = match self.free_slots.pop() {
            Some(idx) => {
                self.entries[idx as usize] = Some(entry);
                idx
            }
            None => {
                self.entries.push(Some(entry));
                (self.entries.len() - 1) as Index
            }
        };

        self.key_map.insert_unique(hash, insert_idx, |&other| {
            self.entries[other as usize]
                .as_ref()
                .expect("slot is present")
                .hash
        });

        self.new_link(insert_idx);
        self.len += 1;
        self.tick();

        None
    }

    /// Removes an entry from the cache with the given key returning the value associated with it
    #[inline]
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hasher.hash_one(key);
        let idx = self.find(key, hash)?;
        Some(self.remove_at_index(idx).1)
    }

    /// Returns a reference to the value for the given key if it exists without bumping the
    /// frequency access count
    #[inline]
    pub fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hasher.hash_one(key);
        let idx = self.find(key, hash)?;
        Some(&self.fetch_entry(idx).value)
    }

    /// Removes the least-frequently used entry from the cache returning the key-value pair
    #[inline]
    pub fn evict(&mut self) -> Option<(K, V)> {
        if self.frequency_head == NULL_IDX {
            return None;
        }

        let target = self.freq_slots[self.frequency_head as usize].tail;
        Some(self.remove_at_index(target))
    }
}

impl<K, V, S: BuildHasher> LfuCache<K, V, S> {
    #[inline]
    fn find<Q>(&self, key: &Q, hash: u64) -> Option<Index>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.key_map
            .find(hash, |&idx| self.fetch_entry(idx).key.borrow() == key)
            .copied()
    }

    #[inline]
    fn fetch_entry(&self, idx: Index) -> &CacheEntry<K, V> {
        self.entries[idx as usize]
            .as_ref()
            .expect("hash implies an entry is present")
    }

    #[inline]
    fn fetch_entry_mut(&mut self, idx: Index) -> &mut CacheEntry<K, V> {
        self.entries[idx as usize]
            .as_mut()
            .expect("hash implies an entry is present")
    }

    #[inline]
    fn is_full(&self) -> bool {
        self.len >= self.capacity
    }

    #[inline]
    fn remove_at_index(&mut self, idx: Index) -> (K, V) {
        self.sever_link(idx);
        let slot_idx = self.fetch_entry(idx).slot_index;
        if self.freq_slots[slot_idx as usize].head == NULL_IDX {
            self.free_freq_slot(slot_idx);
        }

        let hash = self.fetch_entry(idx).hash;
        if let Ok(slot) = self.key_map.find_entry(hash, |&other| other == idx) {
            slot.remove();
        }

        let entry = self.entries[idx as usize]
            .take()
            .expect("value should be present");
        self.free_slots.push(idx);
        self.len -= 1;
        (entry.key, entry.value)
    }

    #[inline]
    fn bump(&mut self, idx: Index) {
        let entry = self.fetch_entry(idx);
        let current_slot = entry.slot_index;
        let current_count = self.freq_slots[current_slot as usize].count;
        self.sever_link(idx);

        let next_slot = self.freq_slots[current_slot as usize].next_slot;
        let next_idx = if next_slot != NULL_IDX
            && self.freq_slots[next_slot as usize].count == current_count + 1
        {
            next_slot
        } else {
            self.insert_freq(current_slot, current_count + 1)
        };

        self.link_front(idx, next_idx);
        if self.freq_slots[current_slot as usize].head == NULL_IDX {
            self.free_freq_slot(current_slot);
        }
    }

    #[inline]
    fn sever_link(&mut self, idx: Index) {
        let entry = self.fetch_entry(idx);
        let (prev, slot, next) = (entry.prev, entry.slot_index, entry.next);

        if prev != NULL_IDX {
            self.fetch_entry_mut(prev).next = next;
        } else {
            self.freq_slots[slot as usize].head = next;
        }

        if next != NULL_IDX {
            self.fetch_entry_mut(next).prev = prev;
        } else {
            self.freq_slots[slot as usize].tail = prev;
        }
    }

    #[inline]
    fn new_link(&mut self, idx: Index) {
        let head = self.frequency_head;
        let new_idx = if head != NULL_IDX && self.freq_slots[head as usize].count == 1 {
            head
        } else {
            self.insert_freq(NULL_IDX, 1)
        };

        self.link_front(idx, new_idx);
    }

    #[inline]
    fn link_front(&mut self, idx: Index, slot_idx: Index) {
        let head = self.freq_slots[slot_idx as usize].head;
        let entry = self.fetch_entry_mut(idx);
        entry.prev = NULL_IDX;
        entry.next = head;
        entry.slot_index = slot_idx;

        if head != NULL_IDX {
            self.fetch_entry_mut(head).prev = idx;
        } else {
            self.freq_slots[slot_idx as usize].tail = idx;
        }

        self.freq_slots[slot_idx as usize].head = idx;
    }

    #[inline]
    fn insert_freq(&mut self, previous: Index, count: u64) -> Index {
        let slot = self.alloc_freq_slot(count);
        let next = if previous != NULL_IDX {
            self.freq_slots[previous as usize].next_slot
        } else {
            self.frequency_head
        };

        self.freq_slots[slot as usize].prev_slot = previous;
        self.freq_slots[slot as usize].next_slot = next;

        if previous != NULL_IDX {
            self.freq_slots[previous as usize].next_slot = slot;
        } else {
            self.frequency_head = slot;
        }

        if next != NULL_IDX {
            self.freq_slots[next as usize].prev_slot = slot;
        }

        slot
    }

    #[inline]
    fn alloc_freq_slot(&mut self, count: u64) -> Index {
        match self.free_freq_slot {
            NULL_IDX => {
                self.freq_slots.push(FreqSlot::new(count));
                (self.freq_slots.len() - 1) as Index
            }
            idx => {
                self.free_freq_slot = self.freq_slots[idx as usize].next_slot;
                self.freq_slots[idx as usize] = FreqSlot::new(count);
                idx
            }
        }
    }

    #[inline]
    fn free_freq_slot(&mut self, idx: Index) {
        let slot = &self.freq_slots[idx as usize];
        let (prev, next) = (slot.prev_slot, slot.next_slot);

        if prev != NULL_IDX {
            self.freq_slots[prev as usize].next_slot = next;
        } else {
            self.frequency_head = next;
        }

        if next != NULL_IDX {
            self.freq_slots[next as usize].prev_slot = prev;
        }

        self.freq_slots[idx as usize].next_slot = self.free_freq_slot;
        self.free_freq_slot = idx;
    }

    #[inline]
    fn tick(&mut self) {
        let Some(interval) = self.age_interval else {
            return;
        };

        self.op_counter += 1;
        if self.op_counter >= interval.get() {
            self.op_counter = 0;
            self.age();
        }
    }

    #[inline]
    fn age(&mut self) {
        let mut slot = self.frequency_head;
        let mut kept = NULL_IDX;

        while slot != NULL_IDX {
            let next = self.freq_slots[slot as usize].next_slot;
            let half = (self.freq_slots[slot as usize].count / 2).max(1);
            self.freq_slots[slot as usize].count = half;

            if kept != NULL_IDX && self.freq_slots[kept as usize].count == half {
                self.merge_after_aging(kept, slot);
                self.free_freq_slot(slot);
            } else {
                kept = slot;
            }

            slot = next;
        }
    }

    #[inline]
    fn merge_after_aging(&mut self, dst: Index, src: Index) {
        let s_head = self.freq_slots[src as usize].head;
        let s_tail = self.freq_slots[src as usize].tail;
        if s_head == NULL_IDX {
            return;
        }

        let mut curr = s_head;
        while curr != NULL_IDX {
            self.fetch_entry_mut(curr).slot_index = dst;
            curr = self.fetch_entry(curr).next;
        }

        let d_head = self.freq_slots[dst as usize].head;
        self.fetch_entry_mut(s_tail).next = d_head;
        if d_head != NULL_IDX {
            self.fetch_entry_mut(d_head).prev = s_tail;
        } else {
            self.freq_slots[dst as usize].tail = s_tail;
        }

        self.freq_slots[dst as usize].head = s_head;
        self.freq_slots[src as usize].head = NULL_IDX;
        self.freq_slots[src as usize].tail = NULL_IDX;
    }
}

impl<K, V, S> Clone for LfuCache<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Clone,
{
    fn clone(&self) -> Self {
        let mut entries = Vec::with_capacity(self.entries.len());

        for entry in self.entries.iter() {
            match entry.as_ref() {
                Some(entry) => {
                    let new = CacheEntry {
                        key: entry.key.clone(),
                        value: entry.value.clone(),
                        hash: entry.hash,
                        prev: entry.prev,
                        next: entry.next,
                        slot_index: entry.slot_index,
                    };
                    entries.push(Some(new));
                }
                None => entries.push(None),
            }
        }

        LfuCache {
            key_map: self.key_map.clone(),
            entries,
            free_slots: self.free_slots.clone(),

            freq_slots: self.freq_slots.clone(),
            free_freq_slot: self.free_freq_slot,
            frequency_head: self.frequency_head,

            len: self.len,
            capacity: self.capacity,
            hasher: self.hasher.clone(),
            op_counter: self.op_counter,
            age_interval: self.age_interval,
        }
    }
}

impl<'a, K, V, S> IntoIterator for &'a LfuCache<K, V, S> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, S> IntoIterator for &'a mut LfuCache<K, V, S> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K, V, S> IntoIterator for LfuCache<K, V, S> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            remaining: self.len,
            slots: self.entries.into_iter(),
        }
    }
}

/// Iterator used for iterating over key-value pairs within the cache
pub struct Iter<'a, K, V> {
    slots: core::slice::Iter<'a, Option<CacheEntry<K, V>>>,
    remaining: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        for slot in self.slots.by_ref() {
            if let Some(entry) = slot {
                self.remaining -= 1;
                return Some((&entry.key, &entry.value));
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, K, V> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while let Some(slot) = self.slots.next_back() {
            if let Some(entry) = slot {
                self.remaining -= 1;
                return Some((&entry.key, &entry.value));
            }
        }

        None
    }
}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {}
impl<K, V> core::iter::FusedIterator for Iter<'_, K, V> {}

/// Mutable iterator for iterating over key-value pairs in the cache
pub struct IterMut<'a, K, V> {
    slots: core::slice::IterMut<'a, Option<CacheEntry<K, V>>>,
    remaining: usize,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        for slot in self.slots.by_ref() {
            if let Some(entry) = slot {
                self.remaining -= 1;
                return Some((&entry.key, &mut entry.value));
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for IterMut<'_, K, V> {}
impl<K, V> core::iter::FusedIterator for IterMut<'_, K, V> {}

/// Wrapper for implementing [`IntoIterator`]
pub struct IntoIter<K, V> {
    slots: std::vec::IntoIter<Option<CacheEntry<K, V>>>,
    remaining: usize,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        for slot in self.slots.by_ref() {
            if let Some(entry) = slot {
                self.remaining -= 1;
                return Some((entry.key, entry.value));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {}
impl<K, V> core::iter::FusedIterator for IntoIter<K, V> {}

impl<K, V> Default for LfuCache<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn puts() {
        let mut lfu = LfuCache::with_capacity(4);
        assert_eq!(lfu.put(1, 1), None);
        assert_eq!(lfu.get(&1), Some(&1));
        assert_eq!(lfu.put(1, 2), Some(1), "put returns the old value");
        assert_eq!(lfu.get(&1), Some(&2));
        assert_eq!(lfu.len(), 1, "overwriting keys does not grow the cache");
    }

    #[test]
    fn puts_low_capacity() {
        let mut lfu = LfuCache::with_capacity(1);
        for i in 0..1000 {
            lfu.put(i, i);
            assert_eq!(lfu.len(), 1);
            assert!(lfu.contains_key(&i));
        }
    }

    #[test]
    fn miss() {
        let mut lfu: LfuCache<u32, u32> = LfuCache::with_capacity(4);
        assert_eq!(lfu.get(&1), None);
        assert_eq!(lfu.get_mut(&1), None);
        assert_eq!(lfu.remove(&1), None);
        assert!(!lfu.contains_key(&1));
        assert!(lfu.is_empty());
    }

    #[test]
    fn get_mut() {
        let mut lfu = LfuCache::with_capacity(4);
        lfu.put(1, 1);
        assert_eq!(lfu.get(&1), Some(&1));
        *lfu.get_mut(&1).expect("valid") = 12;
        assert_eq!(lfu.get(&1), Some(&12));
    }

    #[test]
    fn get_or_insert() {
        let mut lfu = LfuCache::with_capacity(2);
        lfu.put(1, 1);
        let value = lfu.get_or_insert(2, 4);
        assert_eq!(value, &4);
        assert_eq!(lfu.len(), 2);
    }

    #[test]
    fn get_or_insert_with() {
        let mut lfu = LfuCache::with_capacity(2);
        lfu.put(1, 1);
        let value = lfu.get_or_insert_with(2, || 4);
        assert_eq!(value, &4);
        assert_eq!(lfu.len(), 2);
    }

    #[test]
    fn capacity() {
        let empty: LfuCache<u32, u32> = LfuCache::with_capacity(0);
        assert_eq!(empty.capacity(), 1, "capacity is clamped to 1");

        let custom: LfuCache<u32, u32> = LfuCache::with_capacity(8192);
        assert_eq!(custom.capacity(), 8192);

        let default_size: LfuCache<u32, u32> = LfuCache::new();
        assert_eq!(default_size.capacity(), DEFAULT_CAPACITY);
    }

    #[test]
    fn capped_at_capacity() {
        for capacity in [1, 2, 4, 8, 64, 1827] {
            let mut lfu = LfuCache::with_capacity(capacity);
            for i in 0..10_000 {
                lfu.put(i, i);
                assert!(lfu.len() <= capacity, "exceeded capacity");
            }

            assert_eq!(lfu.len(), capacity);
        }
    }

    #[test]
    fn punt_lfu_entry() {
        let mut lfu = LfuCache::with_capacity(4);
        lfu.put(1, 1);
        lfu.put(2, 2);
        lfu.put(3, 3);
        lfu.put(4, 4);

        for _ in 0..10 {
            lfu.get(&1);
            lfu.get(&2);
            lfu.get(&3);
        }

        lfu.put(5, 5);
        assert!(!lfu.contains_key(&4));
        assert_eq!(lfu.get(&1), Some(&1));
        assert_eq!(lfu.get(&2), Some(&2));
        assert_eq!(lfu.get(&3), Some(&3));
        assert_eq!(lfu.get(&5), Some(&5));
    }

    #[test]
    fn punt_lru_of_lfu() {
        let mut lfu = LfuCache::with_capacity(4);
        lfu.put(1, 1);
        lfu.put(2, 2);
        lfu.put(3, 3);
        lfu.put(4, 4);
        lfu.put(5, 5);
        assert!(!lfu.contains_key(&1));
        for i in 2..5 {
            assert!(lfu.contains_key(&i));
        }
    }

    #[test]
    fn update_bumps_frequency() {
        let mut lfu = LfuCache::with_capacity(2);
        lfu.put(1, 1);
        lfu.put(2, 2);
        lfu.put(1, 100);
        lfu.put(3, 3);
        assert!(!lfu.contains_key(&2));
        assert!(lfu.contains_key(&1));
    }

    #[test]
    fn remove_from_middle() {
        let mut lfu = LfuCache::with_capacity(8);
        for i in 0..5 {
            lfu.put(i, i);
        }

        assert_eq!(lfu.remove(&2), Some(2));
        assert_eq!(lfu.len(), 4);

        for i in [0, 1, 3, 4] {
            assert!(lfu.contains_key(&i));
        }
    }

    #[test]
    fn hot_entries() {
        let mut lfu = LfuCache::with_capacity(2);
        lfu.put(1, 1);
        lfu.put(2, 2);

        for _ in 0..100_000 {
            assert_eq!(lfu.get(&1), Some(&1));
        }

        lfu.put(3, 3);
        assert!(!lfu.contains_key(&2));
    }

    #[test]
    fn evict() {
        let mut lfu = LfuCache::with_capacity(2);
        lfu.put(1, 1);
        lfu.put(2, 2);
        lfu.get(&1);
        assert_eq!(lfu.evict(), Some((2, 2)));
        assert_eq!(lfu.len(), 1);
    }

    #[test]
    fn iteration() {
        let mut lfu = LfuCache::with_capacity(4);
        for i in 1..5 {
            lfu.put(i, i);
        }

        let entries: Vec<_> = lfu.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(entries, [(1, 1), (2, 2), (3, 3), (4, 4)]);

        lfu.remove(&2);
        lfu.remove(&4);
        let entries: Vec<_> = lfu.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(entries, [(1, 1), (3, 3)]);
    }
}

struct CacheEntry<K, V> {
    key: K,
    value: V,
    hash: u64,

    // Frequency Indices
    prev: Index,
    next: Index,
    slot_index: Index,
}

#[derive(Clone, Copy)]
struct FreqSlot {
    count: u64,

    // Indicies in the current slot
    head: Index,
    tail: Index,

    // Indicies to the prev and next slots
    prev_slot: Index,
    next_slot: Index,
}

impl FreqSlot {
    pub fn new(count: u64) -> Self {
        Self {
            count,
            head: NULL_IDX,
            tail: NULL_IDX,
            prev_slot: NULL_IDX,
            next_slot: NULL_IDX,
        }
    }
}
