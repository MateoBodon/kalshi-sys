# AWS / SSH Access Notes

## Quick access (SSH)
- Connect: `ssh kalshi-aws`
- Uses `~/.ssh/config` host alias `kalshi-aws`

## Local key + SSH config
- Key path (local): `~/.ssh/kalshi-key.pem` (permissions 600)
- SSH config entry: `~/.ssh/config` host `kalshi-aws`
  - HostName: `98.93.78.177`
  - User: `ubuntu` (switch to `ec2-user` if AMI is Amazon Linux)
  - IdentityFile: `~/.ssh/kalshi-key.pem`
  - StrictHostKeyChecking: `accept-new`
- If you rotate the key, update both the file at `~/.ssh/kalshi-key.pem` and the `IdentityFile` path in `~/.ssh/config`.
