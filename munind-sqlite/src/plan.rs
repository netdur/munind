/// Query plan constants for xBestIndex / xFilter communication.

pub const PLAN_FULLSCAN: i32 = 1;
pub const PLAN_POINT: i32 = 2;
pub const PLAN_KNN: i32 = 3;

// idxStr character codes — each byte encodes the role of an argv parameter.
pub const ARG_MATCH_VECTOR: u8 = b'V';
pub const ARG_K_VALUE: u8 = b'K';
pub const ARG_LIMIT: u8 = b'L';
#[allow(dead_code)]
pub const ARG_ROWID_EQ: u8 = b'R';
