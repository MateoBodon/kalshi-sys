# Index Ladder Windows (ET vs UTC)

Reference windows are defined in `configs/index_ops.yaml`.
Times below assume U.S. **Eastern Time**; convert to UTC for AWS cron/EventBridge.
Cancel buffer is applied ~2 seconds before target.

| Window | Target (ET) | Window Start (ET) | Freeze/Cancel (ET) | Approx UTC start (EST) |
| --- | --- | --- | --- | --- |
| hourly-1000 | 10:00 | 09:45 | 09:59:58 | 14:45
| hourly-1100 | 11:00 | 10:45 | 10:59:58 | 15:45
| hourly-1200 | 12:00 | 11:45 | 11:59:58 | 16:45
| hourly-1300 | 13:00 | 12:45 | 12:59:58 | 17:45
| hourly-1400 | 14:00 | 13:45 | 13:59:58 | 18:45
| hourly-1500 | 15:00 | 14:45 | 14:59:58 | 19:45
| close-1600  | 16:00 | 15:50 | 15:59:58 | 20:50

Notes
- During DST (March–Nov) subtract 4 hours for UTC; during EST subtract 5 hours as above.
- Supervisor logic uses `sched.windows` for the exact bounds and final-minute strictness.
- We trade **INXU/NASDAQ100U** in hourly windows and **INX/NASDAQ100** in the close window.
