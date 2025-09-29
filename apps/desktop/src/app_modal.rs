use std::collections::HashSet;

#[derive(Default, Clone)]
pub(crate) struct PhasePlan {
    pub(crate) sampling: HashSet<String>,
    pub(crate) encoding: HashSet<String>,
}

#[derive(Default, Clone)]
pub(crate) struct PhaseAgg {
    pub(crate) s_cur: u32,
    pub(crate) s_tot: u32,
    pub(crate) e_cur: u32,
    pub(crate) e_tot: u32,
    pub(crate) importing: bool,
    pub(crate) imported: bool,
}
