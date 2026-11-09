#[cfg(not(hifinet))]
mod cesta;
#[cfg(hifinet)]
mod hifinet;

#[cfg(not(hifinet))]
pub(crate) use cesta::{diagnostic, exchange, inference};
#[cfg(hifinet)]
pub(crate) use hifinet::{exchange, inference};
