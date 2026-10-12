/* Linux runtime gate: chroot + dropped identity + Landlock + seccomp.
 * No setuid installation. Parent must already have CAP_SYS_CHROOT (AutoDL root).
 * Landlock ABI >= 1 required; no unrestricted fallback on old kernels.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <stddef.h>
#include <stdint.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef __NR_landlock_create_ruleset
#define __NR_landlock_create_ruleset 444
#define __NR_landlock_add_rule 445
#define __NR_landlock_restrict_self 446
#endif
struct ruleset_attr { uint64_t handled_access_fs; };
struct path_attr { uint64_t allowed_access; int32_t parent_fd; } __attribute__((packed));
#define LL_EXEC (1ULL<<0)
#define LL_WRITE (1ULL<<1)
#define LL_READ (1ULL<<2)
#define LL_DIR (1ULL<<3)
#define LL_ALL_V1 ((1ULL<<13)-1)
#define DENY(nr) BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K, nr, 0, 1), BPF_STMT(BPF_RET|BPF_K, SECCOMP_RET_ERRNO|EPERM)
static void die(const char *s) { perror(s); exit(125); }
static void allow_path(int fd, const char *path, uint64_t flags) {
    int p = open(path, O_PATH|O_CLOEXEC);
    if (p < 0) die("landlock open path");
    struct path_attr a = {.allowed_access=flags,.parent_fd=p};
    if (syscall(__NR_landlock_add_rule,fd,1,&a,0)) die("landlock rule");
    close(p);
}
int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr,"usage: jail ROOT NETWORK MEMORY_BYTES CPU_SECONDS COMMAND...\n"); return 125; }
    int abi = syscall(__NR_landlock_create_ruleset,NULL,0,1);
    if (abi < 1) die("Landlock unsupported");
    uint64_t rights=LL_ALL_V1;
    if (abi>=2) rights|=1ULL<<13; /* REFER */
    if (abi>=3) rights|=1ULL<<14; /* TRUNCATE */
    struct rlimit mem={strtoull(argv[3],NULL,10),strtoull(argv[3],NULL,10)};
    /* CUDA reserves large virtual address ranges. Zero selects no AS ceiling;
       the trusted parent separately monitors resident memory of this group. */
    if (!mem.rlim_cur) mem.rlim_cur=mem.rlim_max=RLIM_INFINITY;
    struct rlimit cpu={strtoul(argv[4],NULL,10),strtoul(argv[4],NULL,10)};
    struct rlimit proc={128,128};
    if (setrlimit(RLIMIT_AS,&mem)||setrlimit(RLIMIT_CPU,&cpu)||setrlimit(RLIMIT_NPROC,&proc)) die("resource limits");
    cpu_set_t available, selected; CPU_ZERO(&selected);
    int want=getenv("R2_CPUS")?atoi(getenv("R2_CPUS")):2, used=0;
    if (want<1 || sched_getaffinity(0,sizeof(available),&available)) die("CPU affinity");
    for (int i=0;i<CPU_SETSIZE&&used<want;i++) if (CPU_ISSET(i,&available)) { CPU_SET(i,&selected);used++; }
    if (sched_setaffinity(0,sizeof(selected),&selected)) die("set CPU affinity");
    if (chdir(argv[1])||chroot(".")||chdir("/code")) die("chroot unavailable");
    if (setgroups(0,NULL)||setgid(65534)||setuid(65534)) die("drop identity");
    if (prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0)) die("no_new_privs");
    struct ruleset_attr r={.handled_access_fs=rights};
    int fd=syscall(__NR_landlock_create_ruleset,&r,sizeof(r),0);
    if (fd<0) die("landlock create");
    allow_path(fd,"/",LL_EXEC|LL_READ|LL_DIR);
    allow_path(fd,"/out",rights);
    allow_path(fd,"/tmp",rights);
    /* /dev/null and /dev/urandom are the only writable standard device handles. */
    allow_path(fd,"/dev/null",LL_READ|LL_WRITE);
    allow_path(fd,"/dev",LL_READ|LL_DIR|LL_WRITE);
    if (syscall(__NR_landlock_restrict_self,fd,0)) die("landlock enforce");
    close(fd);
    struct sock_filter filter[]={
      BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,arch)),
      BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,AUDIT_ARCH_X86_64,1,0),
      BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_KILL_PROCESS),
      BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,nr)),
      /* Reject x32 ABI as well as a different audit architecture. */
      BPF_JUMP(BPF_JMP|BPF_JGE|BPF_K,0x40000000,0,1),
      BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_KILL_PROCESS),
      DENY(__NR_ptrace), DENY(__NR_process_vm_readv), DENY(__NR_process_vm_writev),
      DENY(__NR_mount), DENY(__NR_umount2), DENY(__NR_pivot_root), DENY(__NR_chroot),
      DENY(__NR_unshare), DENY(__NR_setns), DENY(__NR_bpf), DENY(__NR_userfaultfd),
      DENY(__NR_open_by_handle_at), DENY(__NR_perf_event_open), DENY(__NR_io_uring_setup),
      DENY(__NR_keyctl), DENY(__NR_add_key), DENY(__NR_request_key),
      DENY(__NR_setsid), DENY(__NR_setpgid), DENY(__NR_setrlimit),
      DENY(__NR_kill), DENY(__NR_tkill),
      /* Network rule is patched to an impossible syscall when explicitly staging packages. */
      DENY(0x7ffffffe),
      BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW)
    };
    if (argv[2][0]!='1') filter[(sizeof(filter)/sizeof(filter[0]))-3].k=__NR_socket;
    struct sock_fprog program={.len=sizeof(filter)/sizeof(filter[0]),.filter=filter};
    if (prctl(PR_SET_SECCOMP,SECCOMP_MODE_FILTER,&program)) die("seccomp enforce");
    execv(argv[5],argv+5); die("exec");
}
