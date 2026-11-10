#[cfg(not(any(hifinet, stgcn)))]
mod cesta;
#[cfg(hifinet)]
mod hifinet;

#[cfg(not(any(hifinet, stgcn)))]
pub(crate) use cesta::{diagnostic, exchange, inference};
#[cfg(hifinet)]
pub(crate) use hifinet::{exchange, inference};

#[cfg(stgcn)]
mod stgcn;
#[cfg(stgcn)]
pub(crate) use stgcn::{exchange, inference};
