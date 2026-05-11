# guldNAS Inspection — 2026-05-11

This file contains the raw terminal inspection output from `guldNAS` before creating the StockInvestmentDSS canonical storage structure.

## Summary

- Hostname: `guldNAS`
- User: `guldNAS`
- Home: `/home/guldNAS`
- RAID device: `/dev/md0`
- RAID level: RAID1
- RAID members: `/dev/sda`, `/dev/sdb`
- RAID state: clean, `[UU]`, 2/2 active devices
- Filesystem: ext4
- Mount point: `/mnt/nas`
- Samba share: `[nas]`
- Samba path: `/mnt/nas`
- Samba user: `guldNAS`
- IP address: `10.0.0.164`

## Raw Output

```text
guldNAS@guldNAS:~$ echo "============================================================"
echo "BASIC SYSTEM INFO"
echo "============================================================"
hostname
whoami
pwd
date
uname -a

echo ""
echo "============================================================"
echo "DISK / FILESYSTEM OVERVIEW"
echo "============================================================"
df -h
echo ""
lsblk -f
echo ""
lsblk -o NAME,SIZE,FSTYPE,TYPE,MOUNTPOINTS,UUID

echo ""
echo "============================================================"
echo "COMMON STORAGE LOCATIONS"
echo "============================================================"
echo "--- /mnt ---"
ls -la /mnt 2>/dev/null || true

echo ""
echo "--- /media ---"
ls -la /media 2>/dev/null || true

echo ""
echo "--- /srv ---"
ls -la /srv 2>/dev/null || true

echo ""
echo "--- /home ---"
ls -la /home 2>/dev/null || true

echo ""
echo "--- current home ---"
echo "============================================================"only|writable|valid users|guest ok|force user|create mask|directory mask' || true
============================================================
BASIC SYSTEM INFO
============================================================
guldNAS
guldNAS
/home/guldNAS
Mon May 11 02:01:59 CEST 2026
Linux guldNAS 5.15.0-1098-raspi #101-Ubuntu SMP PREEMPT Thu Mar 19 11:01:01 UTC 2026 aarch64 aarch64 aarch64 GNU/Linux

============================================================
DISK / FILESYSTEM OVERVIEW
============================================================
Filesystem      Size  Used Avail Use% Mounted on
tmpfs           781M  4.9M  776M   1% /run
/dev/mmcblk0p2   30G  3.9G   24G  15% /
tmpfs           3.9G     0  3.9G   0% /dev/shm
tmpfs           5.0M     0  5.0M   0% /run/lock
/dev/mmcblk0p1  253M  146M  107M  58% /boot/firmware
log2ram         256M   37M  220M  15% /var/log
tmpfs           781M  4.0K  781M   1% /run/user/1000
/dev/md0        916G   44K  870G   1% /mnt/nas

NAME        FSTYPE            FSVER LABEL       UUID                                 FSAVAIL FSUSE% MOUNTPOINTS
loop0       squashfs          4.0                                                          0   100% /snap/core20/2720
loop1       squashfs          4.0                                                          0   100% /snap/core20/2772
loop2       squashfs          4.0                                                          0   100% /snap/lxd/38472
loop3       squashfs          4.0                                                          0   100% /snap/lxd/38801
loop4       squashfs          4.0                                                          0   100% /snap/snapd/26383
loop5       squashfs          4.0                                                          0   100% /snap/snapd/26869
sda         linux_raid_member 1.2   guldNAS:0   62fa3465-b3bc-3654-4745-f992c730132d
└─md0       ext4              1.0               0e962813-0ee8-425f-b932-5983446739aa  869.1G     0% /mnt/nas
sdb         linux_raid_member 1.2   guldNAS:0   62fa3465-b3bc-3654-4745-f992c730132d
└─md0       ext4              1.0               0e962813-0ee8-425f-b932-5983446739aa  869.1G     0% /mnt/nas
mmcblk0
├─mmcblk0p1 vfat              FAT32 system-boot B1C2-332F                             106.9M    58% /boot/firmware
└─mmcblk0p2 ext4              1.0   writable    3b687fda-fd50-40d7-9407-5f6103485cda   23.9G    13% /var/hdd.log
                                                                                                    /

NAME          SIZE FSTYPE            TYPE  MOUNTPOINTS       UUID
loop0        59.6M squashfs          loop  /snap/core20/2720
loop1        59.6M squashfs          loop  /snap/core20/2772
loop2          81M squashfs          loop  /snap/lxd/38472
loop3          81M squashfs          loop  /snap/lxd/38801
loop4        41.8M squashfs          loop  /snap/snapd/26383
loop5        42.6M squashfs          loop  /snap/snapd/26869
sda         931.5G linux_raid_member disk                    62fa3465-b3bc-3654-4745-f992c730132d
└─md0       931.4G ext4              raid1 /mnt/nas          0e962813-0ee8-425f-b932-5983446739aa
sdb         931.5G linux_raid_member disk                    62fa3465-b3bc-3654-4745-f992c730132d
└─md0       931.4G ext4              raid1 /mnt/nas          0e962813-0ee8-425f-b932-5983446739aa
mmcblk0      29.8G                   disk
├─mmcblk0p1   256M vfat              part  /boot/firmware    B1C2-332F
└─mmcblk0p2  29.6G ext4              part  /var/hdd.log      3b687fda-fd50-40d7-9407-5f6103485cda
                                           /

============================================================
COMMON STORAGE LOCATIONS
============================================================
--- /mnt ---
total 12
drwxr-xr-x  3 root    root    4096 May  1 16:18 .
drwxr-xr-x 20 root    root    4096 May  1 15:42 ..
drwxr-xr-x  3 guldNAS guldNAS 4096 May  1 16:52 nas

--- /media ---
total 8
drwxr-xr-x  2 root root 4096 Feb 17  2023 .
drwxr-xr-x 20 root root 4096 May  1 15:42 ..

--- /srv ---
total 8
drwxr-xr-x  2 root root 4096 Feb 17  2023 .
drwxr-xr-x 20 root root 4096 May  1 15:42 ..

--- /home ---
total 12
drwxr-xr-x  3 root    root    4096 Feb 17  2023 .
drwxr-xr-x 20 root    root    4096 May  1 15:42 ..
drwxr-x---  6 guldNAS guldNAS 4096 May 10 11:54 guldNAS

--- current home ---
total 48
drwxr-x--- 6 guldNAS guldNAS 4096 May 10 11:54 .
drwxr-xr-x 3 root    root    4096 Feb 17  2023 ..
-rw------- 1 guldNAS guldNAS 2473 May  1 15:42 .bash_history
-rw-r--r-- 1 guldNAS guldNAS  220 Jan  6  2022 .bash_logout
-rw-r--r-- 1 guldNAS guldNAS 3771 Jan  6  2022 .bashrc
drwx------ 2 guldNAS guldNAS 4096 Mar 12  2023 .cache
drwx------ 3 guldNAS guldNAS 4096 Dec 23  2023 .config
-rw------- 1 guldNAS guldNAS   20 May 10 11:54 .lesshst
drwxrwxr-x 3 guldNAS guldNAS 4096 Dec 23  2023 .local
-rw-r--r-- 1 guldNAS guldNAS  807 Jan  6  2022 .profile
drwx------ 2 guldNAS guldNAS 4096 May  1 19:56 .ssh
-rw-r--r-- 1 guldNAS guldNAS    0 Mar 12  2023 .sudo_as_admin_successful
-rw-rw-r-- 1 guldNAS guldNAS  163 Mar 12  2023 .wget-hsts

============================================================
FIND EXISTING STORAGE-LIKE DIRECTORIES
============================================================
/home
/home/guldNAS
/home/guldNAS/.cache
/home/guldNAS/.config
/home/guldNAS/.config/procps
/home/guldNAS/.local
/home/guldNAS/.local/share
/home/guldNAS/.ssh
/media
/mnt
/mnt/nas
/mnt/nas/lost+found
/srv

============================================================
RAID / MDADM STATUS
============================================================
Personalities : [linear] [multipath] [raid0] [raid1] [raid6] [raid5] [raid4] [raid10]
md0 : active raid1 sdb[1] sda[0]
      976630464 blocks super 1.2 [2/2] [UU]
      bitmap: 0/8 pages [0KB], 65536KB chunk

unused devices: <none>

[sudo] password for guldNAS:
ARRAY /dev/md0 metadata=1.2 name=guldNAS:0 UUID=62fa3465:b3bc3654:4745f992:c730132d

/dev/md0:
           Version : 1.2
     Creation Time : Fri May  1 16:16:25 2026
        Raid Level : raid1
        Array Size : 976630464 (931.39 GiB 1000.07 GB)
     Used Dev Size : 976630464 (931.39 GiB 1000.07 GB)
      Raid Devices : 2
     Total Devices : 2
       Persistence : Superblock is persistent

     Intent Bitmap : Internal

       Update Time : Mon May 11 02:02:10 2026
             State : clean
    Active Devices : 2
   Working Devices : 2
    Failed Devices : 0
     Spare Devices : 0

Consistency Policy : bitmap

              Name : guldNAS:0  (local to host guldNAS)
              UUID : 62fa3465:b3bc3654:4745f992:c730132d
            Events : 1174

    Number   Major   Minor   RaidDevice State
       0       8        0        0      active sync   /dev/sda
       1       8       16        1      active sync   /dev/sdb

============================================================
FSTAB
============================================================
LABEL=writable  /       ext4    discard,errors=remount-ro       0 1
LABEL=system-boot       /boot/firmware  vfat    defaults        0       1
UUID=0e962813-0ee8-425f-b932-5983446739aa /mnt/nas ext4 defaults,nofail,noatime 0 2

============================================================
SAMBA STATUS
============================================================
● smbd.service - Samba SMB Daemon
     Loaded: loaded (/lib/systemd/system/smbd.service; enabled; vendor preset: enabled)
     Active: active (running) since Fri 2026-05-01 16:43:12 CEST; 1 week 2 days ago
       Docs: man:smbd(8)
             man:samba(7)
             man:smb.conf(5)
    Process: 12898 ExecStartPre=/usr/share/samba/update-apparmor-samba-profile (code=exited, status=0/SUCCESS)
   Main PID: 12907 (smbd)
     Status: "smbd: ready to serve connections..."
      Tasks: 4 (limit: 9240)
     Memory: 9.5M
        CPU: 49.927s
     CGroup: /system.slice/smbd.service
             ├─12907 /usr/sbin/smbd --foreground --no-process-group
             ├─12909 /usr/sbin/smbd --foreground --no-process-group
             ├─12910 /usr/sbin/smbd --foreground --no-process-group
             └─12911 /usr/lib/aarch64-linux-gnu/samba/samba-bgqd --ready-signal-fd=45 --parent-watch-fd=11 --debuglevel=0 -F

May 01 16:43:12 guldNAS systemd[1]: Starting Samba SMB Daemon...
May 01 16:43:12 guldNAS systemd[1]: Started Samba SMB Daemon.
May 01 16:52:22 guldNAS smbd[13464]: pam_unix(samba:session): session closed for user nobody
May 01 16:52:22 guldNAS smbd[13464]: pam_unix(samba:session): session closed for user nobody
May 01 16:52:42 guldNAS smbd[13464]: pam_unix(samba:session): session opened for user guldNAS(uid=1000) by (uid=0)
May 03 20:06:53 guldNAS smbd[13464]: pam_unix(samba:session): session closed for user guldNAS
May 03 20:06:53 guldNAS smbd[13464]: pam_unix(samba:session): session opened for user guldNAS(uid=1000) by (uid=0)
May 03 20:19:02 guldNAS smbd[13464]: pam_unix(samba:session): session closed for user guldNAS
May 03 20:19:02 guldNAS smbd[13464]: pam_unix(samba:session): session opened for user guldNAS(uid=1000) by (uid=0)
May 03 20:19:02 guldNAS smbd[13464]: pam_unix(samba:session): session closed for user guldNAS

============================================================
SAMBA CONFIG SUMMARY
============================================================
[global]
[printers]
        browseable = No
        create mask = 0700
        path = /var/spool/samba
[print$]
        path = /var/lib/samba/printers
[nas]
        path = /mnt/nas
        read only = No
        valid users = guldNAS

============================================================
FULL SAMBA SHARES FROM CONFIG
============================================================
# Global parameters
[global]
        log file = /var/log/samba/log.%m
        logging = file
        map to guest = Bad User
        max log size = 1000
        obey pam restrictions = Yes
        pam password change = Yes
        panic action = /usr/share/samba/panic-action %d
        passwd chat = *Enter\snew\s*\spassword:* %n\n *Retype\snew\s*\spassword:* %n\n *password\supdated\ssuccessfully* .
        passwd program = /usr/bin/passwd %u
        server role = standalone server
        server string = %h server (Samba, Ubuntu)
        unix password sync = Yes
        usershare allow guests = Yes
        idmap config * : backend = tdb


[printers]
        browseable = No
        comment = All Printers
        create mask = 0700
        path = /var/spool/samba
        printable = Yes


[print$]
        comment = Printer Drivers
        path = /var/lib/samba/printers


[nas]
        path = /mnt/nas
        read only = No
        valid users = guldNAS

============================================================
NETWORK INFO
============================================================
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    inet 127.0.0.1/8 scope host lo
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    inet 10.0.0.164/24 metric 100 brd 10.0.0.255 scope global dynamic eth0

10.0.0.164

============================================================
DONE
============================================================
guldNAS@guldNAS:~$
```
