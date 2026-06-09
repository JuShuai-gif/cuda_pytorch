# Real-Time Scheduling Simulator

Simulates RMS (Rate Monotonic Scheduling) and EDF (Earliest Deadline First)
scheduling algorithms with priority inversion/inheritance demonstrations.

## Features

- **RMS Scheduler**: Fixed-priority scheduling, shorter period = higher priority
- **EDF Scheduler**: Dynamic-priority scheduling, nearest deadline = highest priority
- **Priority Inversion Demo**: Classic 3-task scenario with OS threads
- **Priority Inheritance Demo**: Solution with simulated priority inheritance mutex
- **Schedule Timeline Visualization**: ASCII art timeline of task execution
- **Deadline Miss Detection**: Reports missed deadlines with violation amounts

## File Structure

```
12_realtime_system/
|-- task.h                  # Task struct, ScheduleEvent, SchedulerStats data types
|-- rms_scheduler.h         # Scheduler base class + RMSScheduler declaration
|-- rms_scheduler.cpp       # RMSScheduler implementation (fixed-priority scheduling)
|-- edf_scheduler.h         # EDFScheduler declaration
|-- edf_scheduler.cpp       # EDFScheduler implementation (dynamic-priority scheduling)
|-- priority_inversion.h    # SharedResource, PriorityInheritanceMutex, demo declarations
|-- priority_inversion.cpp  # Priority inversion + inheritance demo implementations
|-- main.cpp                # CLI parsing, parse_tasks, print_timeline, entry point
|-- CMakeLists.txt
|-- README.md
```

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

```bash
# Run both RMS and EDF with default task set + inversion/inheritance demos
./rt_sim

# Run only RMS
./rt_sim --mode rms

# Run only EDF
./rt_sim --mode edf

# Custom task set: name:C:T:D (C=WCET, T=period, D=deadline)
./rt_sim --tasks "A:1000:3000:3000,B:1500:5000:5000,C:2000:8000:8000"

# With schedule timeline
./rt_sim --timeline

# Run priority inversion demo only
./rt_sim --demo-inversion

# Run priority inheritance demo only
./rt_sim --demo-inheritance

# High-resolution simulation (100us tick)
./rt_sim --tick 100
```

## Default Task Set

```
Task A: C=1ms, T=4ms, D=4ms
Task B: C=2ms, T=6ms, D=6ms
Task C: C=3ms, T=12ms, D=12ms
```

Utilization: 1/4 + 2/6 + 3/12 = 83.3% (exceeds RMS bound of 78%, but EDF can handle it)

## Sample Output

```
============================================================
  Real-Time Scheduling Simulator
  RMS + EDF + Priority Inversion Demos
============================================================

=== RMS (Rate Monotonic Scheduling) ===
Hyperperiod: 12 ms
Task          C(us)   T(us)   D(us)   Priority
----------------------------------------------
A             1000    4000    4000    0
B             2000    6000    6000    1
C             3000    12000   12000   2

RMS Result:
  Total jobs:       6
  Missed deadlines: 1
  Preemptions:      5
  Context switches: 12
  Avg response:     2.8 ms
  Max response:     3.0 ms
  Total utilization: 83.33%
  RMS bound (n=3): 77.98%
  Status: DEADLINE MISSES DETECTED

=== EDF (Earliest Deadline First) ===
Hyperperiod: 12 ms
...

EDF Result:
  Total jobs:       6
  Missed deadlines: 0
  Total utilization: 83.33%
  Status: SCHEDULABLE (EDF bound is 100%)
```

## Priority Inversion Demo

Demonstrates the classic Mars Pathfinder scenario:
- Task H (high priority) waits for a lock held by Task L (low priority)
- Task M (medium priority, no lock needed) preempts Task L indefinitely
- Task H is blocked for the duration of Task M's execution (unbounded)

## Priority Inheritance Demo

Shows the solution:
- Task L inherits Task H's priority while holding the lock
- Task M cannot preempt Task L
- Task H's blocking time is bounded by the critical section length
