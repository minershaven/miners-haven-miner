//! Windows "large pages" (huge pages) for the RandomX dataset: about 10-20% more hashrate.
//!
//! Windows only hands out large pages to accounts holding the "Lock pages in memory" right
//! (SeLockMemoryPrivilege). When the miner runs as administrator it grants that right to the
//! current user itself; Windows applies it at the next sign-in, after which the miner enables it
//! on every start. For administrator accounts Windows strips the right from non-elevated
//! programs, so the miner has to be run as administrator to use it.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The right is held and enabled: the dataset can use large pages.
    Active,
    /// The right was just granted; sign out and back in (or restart) once to activate it.
    GrantedSignOutNeeded,
    /// Not running as administrator, so the right can't be granted or used.
    NeedsAdmin,
    /// Running as administrator but granting failed (e.g. blocked by group policy).
    Failed,
    /// Not a Windows build (Linux uses the system's hugepages setting).
    NotWindows,
}

impl Status {
    pub fn describe(self) -> &'static str {
        match self {
            Status::Active => "huge pages enabled",
            Status::GrantedSignOutNeeded => "huge pages permission granted - sign out of Windows and back in (or restart) once to activate",
            Status::NeedsAdmin => "huge pages off - run as administrator to use them (about 10-20% faster)",
            Status::Failed => "huge pages could not be enabled on this PC (blocked by policy?)",
            Status::NotWindows => "huge pages: uses the system setting",
        }
    }
}

#[cfg(not(windows))]
pub fn setup() -> Status {
    Status::NotWindows
}

#[cfg(not(windows))]
pub fn is_admin() -> bool {
    false
}

#[cfg(not(windows))]
pub fn relaunch_as_admin(_args: &str) -> bool {
    false
}

#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    use std::mem::{size_of, zeroed};
    use std::ptr::{null, null_mut};
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_NOT_ALL_ASSIGNED, HANDLE};
    use windows_sys::Win32::Security::Authentication::Identity::{
        LsaAddAccountRights, LsaClose, LsaOpenPolicy, LSA_OBJECT_ATTRIBUTES, LSA_UNICODE_STRING,
        POLICY_CREATE_ACCOUNT, POLICY_LOOKUP_NAMES,
    };
    use windows_sys::Win32::Security::{
        AdjustTokenPrivileges, GetTokenInformation, LookupPrivilegeValueW, TokenElevation, TokenUser,
        LUID_AND_ATTRIBUTES, SE_PRIVILEGE_ENABLED, TOKEN_ADJUST_PRIVILEGES, TOKEN_ELEVATION, TOKEN_PRIVILEGES,
        TOKEN_QUERY, TOKEN_USER,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};
    use windows_sys::Win32::UI::Shell::ShellExecuteW;

    use super::Status;

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    struct Token(HANDLE);
    impl Drop for Token {
        fn drop(&mut self) {
            unsafe { CloseHandle(self.0) };
        }
    }

    fn token() -> Option<Token> {
        let mut h: HANDLE = null_mut();
        let ok = unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &mut h) };
        if ok == 0 { None } else { Some(Token(h)) }
    }

    pub fn is_admin() -> bool {
        let Some(t) = token() else { return false };
        let mut e: TOKEN_ELEVATION = unsafe { zeroed() };
        let mut len = 0u32;
        let ok = unsafe {
            GetTokenInformation(t.0, TokenElevation, &mut e as *mut _ as *mut c_void, size_of::<TOKEN_ELEVATION>() as u32, &mut len)
        };
        ok != 0 && e.TokenIsElevated != 0
    }

    /// Turn on SeLockMemoryPrivilege in our token. Fails unless the account holds the right.
    fn enable_privilege() -> bool {
        let Some(t) = token() else { return false };
        let name = wide("SeLockMemoryPrivilege");
        let mut tp: TOKEN_PRIVILEGES = unsafe { zeroed() };
        tp.PrivilegeCount = 1;
        tp.Privileges = [LUID_AND_ATTRIBUTES { Luid: unsafe { zeroed() }, Attributes: SE_PRIVILEGE_ENABLED }];
        if unsafe { LookupPrivilegeValueW(null(), name.as_ptr(), &mut tp.Privileges[0].Luid) } == 0 {
            return false;
        }
        let ok = unsafe { AdjustTokenPrivileges(t.0, 0, &tp, 0, null_mut(), null_mut()) };
        ok != 0 && unsafe { GetLastError() } != ERROR_NOT_ALL_ASSIGNED
    }

    /// Give the current user the "Lock pages in memory" right (needs administrator).
    fn grant_right() -> bool {
        let Some(t) = token() else { return false };
        let mut len = 0u32;
        unsafe { GetTokenInformation(t.0, TokenUser, null_mut(), 0, &mut len) };
        if len == 0 {
            return false;
        }
        let mut buf = vec![0u8; len as usize];
        if unsafe { GetTokenInformation(t.0, TokenUser, buf.as_mut_ptr() as *mut c_void, len, &mut len) } == 0 {
            return false;
        }
        let user = unsafe { &*(buf.as_ptr() as *const TOKEN_USER) };
        let sid = user.User.Sid;

        let attrs: LSA_OBJECT_ATTRIBUTES = unsafe { zeroed() };
        let mut policy: *mut c_void = null_mut();
        let st = unsafe { LsaOpenPolicy(null(), &attrs, (POLICY_CREATE_ACCOUNT | POLICY_LOOKUP_NAMES) as u32, &mut policy as *mut _ as _) };
        if st != 0 {
            return false;
        }
        let mut name = wide("SeLockMemoryPrivilege");
        let chars = (name.len() - 1) as u16;
        let right = LSA_UNICODE_STRING { Length: chars * 2, MaximumLength: chars * 2 + 2, Buffer: name.as_mut_ptr() };
        let st = unsafe { LsaAddAccountRights(policy as _, sid, &right, 1) };
        unsafe { LsaClose(policy as _) };
        st == 0
    }

    pub fn setup() -> Status {
        if enable_privilege() {
            return Status::Active;
        }
        if !is_admin() {
            return Status::NeedsAdmin;
        }
        if grant_right() {
            // Windows only applies a newly granted right at the next sign-in
            if enable_privilege() { Status::Active } else { Status::GrantedSignOutNeeded }
        } else {
            Status::Failed
        }
    }

    /// Start this program again with a UAC prompt ("Run as administrator").
    pub fn relaunch_as_admin(args: &str) -> bool {
        let Ok(exe) = std::env::current_exe() else { return false };
        let exe = wide(&exe.to_string_lossy());
        let verb = wide("runas");
        let params = wide(args);
        let r = unsafe { ShellExecuteW(null_mut(), verb.as_ptr(), exe.as_ptr(), params.as_ptr(), null(), 1) };
        r as usize > 32
    }
}

#[cfg(windows)]
pub use win::{is_admin, relaunch_as_admin, setup};
