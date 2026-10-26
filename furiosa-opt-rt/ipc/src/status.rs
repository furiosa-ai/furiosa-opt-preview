/// Why a request failed or a task exited unsuccessfully. Zero is not a status: an [`Answer`](crate::Answer)
/// encodes success as zero, and a task exits with zero when it succeeds. Retired wire values stay
/// reserved.
#[derive(thiserror::Error, Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Status {
    #[error("the request, arguments or copy could not be accepted")]
    BadArgs = 1,
    /// The response carries the task's panic message.
    #[error("the task panicked")]
    Panicked = 3,
    #[error("the launch ran past its time limit")]
    TimedOut = 4,
    #[error("the cluster was never initialized")]
    NoIdentity = 6,
    /// Includes unusable images, code-transfer failures and unknown task exit/yield codes.
    #[error("the image or task execution could not be used")]
    BadImage = 7,
    /// A peer outside the cluster, or a ring whose index register holds nonsense.
    #[error("the task made a syscall the firmware could not serve")]
    BadSyscall = 8,
    #[error("the firmware could not configure DMA, serve a task event or encode its response")]
    Internal = 9,
}

impl TryFrom<i32> for Status {
    type Error = i32;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::BadArgs),
            3 => Ok(Self::Panicked),
            4 => Ok(Self::TimedOut),
            6 => Ok(Self::NoIdentity),
            7 => Ok(Self::BadImage),
            8 => Ok(Self::BadSyscall),
            9 => Ok(Self::Internal),
            other => Err(other),
        }
    }
}

impl From<Status> for i32 {
    fn from(status: Status) -> Self {
        status as Self
    }
}

#[cfg(test)]
mod tests {
    use super::Status;

    /// Task-library exit codes keep their wire meaning and do not accept reserved values.
    #[test]
    fn preserves_exit_codes() {
        for (value, status) in [
            (1, Status::BadArgs),
            (3, Status::Panicked),
            (4, Status::TimedOut),
            (6, Status::NoIdentity),
            (7, Status::BadImage),
            (8, Status::BadSyscall),
            (9, Status::Internal),
        ] {
            assert_eq!(Status::try_from(value), Ok(status));
            assert_eq!(i32::from(status), value);
        }
        for value in [i32::MIN, -1, 0, 2, 5, 10, i32::MAX] {
            assert_eq!(Status::try_from(value), Err(value));
        }
    }
}
