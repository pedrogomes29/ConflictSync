use std::{collections::hash_map::Keys, hash::Hash};

use anyhow::ensure;
use fxhash::FxHashMap;

use crate::{
    dot_context::{Delta as CtxDelta, DotKind},
    Decompose, Dot, DotContext, Extract,
};

#[derive(Clone, Debug, Default)]
pub struct AWSet<I, T> {
    elems: FxHashMap<T, Dot<I>>,
    ctx: DotContext<I>,
}

#[derive(Clone, Debug)]
pub struct Delta<'a, I, T> {
    set: &'a FxHashMap<T, Dot<I>>,
    elems: Vec<(&'a T, &'a Dot<I>)>,
    ctx: CtxDelta<'a, I>,
}

impl<I, T> AWSet<I, T> {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            elems: FxHashMap::default(),
            ctx: DotContext::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.elems.is_empty()
    }

    pub fn len(&self) -> usize {
        self.elems.len()
    }

    pub fn iter(&self) -> Keys<'_, T, Dot<I>> {
        self.elems.keys()
    }
}

impl<I, T> AWSet<I, T>
where
    T: Eq + Hash,
{
    pub fn contains(&self, value: &T) -> bool {
        self.elems.contains_key(value)
    }
}

impl<I, T> AWSet<I, T>
where
    I: Clone + Eq + Hash,
    T: Clone + Eq + Hash,
{
    pub fn insert(&mut self, id: &I, value: T) -> Delta<'_, I, T> {
        match self.elems.get_mut(&value) {
            Some(Dot(i, n)) if i == id => *n = self.ctx.max(id) + 1,
            _ => {
                self.elems
                    .insert(value.clone(), Dot(id.clone(), self.ctx.max(id) + 1));
            }
        }

        let entry = self
            .elems
            .get_key_value(&value)
            .expect("value should exist at this point");

        Delta {
            set: &self.elems,
            elems: vec![entry],
            ctx: self.ctx.next(id),
        }
    }

    pub fn remove(&mut self, value: &T) -> Delta<'_, I, T> {
        self.elems.remove(value);

        Delta {
            set: &self.elems,
            elems: vec![],
            ctx: CtxDelta::empty_with(&self.ctx),
        }
    }
}

impl<I, T> PartialEq for AWSet<I, T>
where
    T: Eq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.elems.keys().all(|v| other.elems.contains_key(v))
    }
}

impl<I, T> From<Delta<'_, I, T>> for AWSet<I, T>
where
    I: Clone + Eq + Ord + Hash,
    T: Clone + Eq + Hash,
{
    fn from(value: Delta<'_, I, T>) -> Self {
        Self {
            elems: value
                .elems
                .into_iter()
                .map(|(i, n)| (i.clone(), n.clone()))
                .collect(),
            ctx: DotContext::from(value.ctx),
        }
    }
}

impl<I, T> Decompose for AWSet<I, T>
where
    I: Clone + Eq + Ord + Hash,
    T: Clone + Eq + Hash,
{
    type Decomposition<'a> = Delta<'a, I, T> where I: 'a, T: 'a;

    fn split(&self) -> Vec<Self::Decomposition<'_>> {
        let elements = self.elems.iter().map(|entry| Delta {
            set: &self.elems,
            elems: vec![entry],
            ctx: CtxDelta::empty_with(&self.ctx),
        });

        let context = self.ctx.split().into_iter().map(|d| Delta {
            set: &self.elems,
            elems: vec![],
            ctx: d,
        });

        elements.chain(context).collect()
    }

    fn join(&mut self, _deltas: Vec<Self::Decomposition<'_>>) {
        todo!()
    }

    fn difference<'a>(&'a self, _remote: &'a Self) -> Self::Decomposition<'a> {
        todo!()
    }
}

#[derive(Clone, Debug, Hash)]
pub enum Element<'a, I, T> {
    Entry(&'a T, &'a Dot<I>),
    Context(DotKind<'a, I>),
}

impl<'a, I, T> Extract for Delta<'a, I, T>
where
    I: Hash,
    T: Hash,
{
    type Output = Element<'a, I, T>;

    fn extract(&self) -> anyhow::Result<Self::Output> {
        let ctx_extraction = self.ctx.extract().map(|d| Element::<I, T>::Context(d));
        let values = self.elems.len() + ctx_extraction.is_ok() as usize;
        ensure!(
            values == 1,
            "decomposition should contain a single value, but instead got {values} values"
        );

        match self.elems.first() {
            Some((v, d)) => Ok(Element::Entry(v, d)),
            None => unreachable!("decomposition contains at least a value"),
        }
    }
}
