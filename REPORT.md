# Assignment 1 Report

- **Name**: Arturo Velilla Martinez
- **Student ID**: 19455
- **Email**: avelilla.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

The model uses the US Bureau of Transportation Statistics Reporting Carrier On-Time Performance table for March 2026 which has 612,102 domestic flights and 17 columns. It's committed as `data/march2026.zip`. One modelling row is one flight scheduled to arrive at Detroit Metropolitan Airport (DTW).

**Why this dataset.** It supports a real operational decision, it is public domain so it can be committed and retrained by anyone, and it is raw operational data.

**Why DTW.** It is a connecting hub, so protecting connections is a real problem there, and one month of arrivals, about 10.7k rows, fits the ~20k-row guide. I first tried Dallas/Fort Worth, but it was too large, then with Austin, which is smaller but is not a connecting hub.

**Why March 2026.** I compared it with October 2025 which had 13.8% late and almost no cancellations. October would have given a cleaner split, but it barely contains any disruption, which is exactly what the business case is about. March contained a weather disruption on 13-18 March that pushed DTW to 57-64% late on its worst days.

**Why the national file and not only DTW rows.** The strongest features describe the aircraft's previous leg and how other airports are doing that day; those rows are not DTW arrivals. The pipeline therefore builds features on all 612k flights and only then keeps DTW arrivals. The zipped national file is small enough to commit.

**Supporting file.** Bureau of Transportation Statistics times are local to each airport. `data/airport_timezones.csv` maps all 342 airports in the file to their own time zone, so every time can be converted to UTC; it was generated once from the `airportsdata` package and committed as data, to avoid adding a dependency.

## Business / real-life framing

**Decision.** DTW's hub operations centre decides, flight by flight, whether to activate connection protection for an inbound flight: alert re-accommodation staff, pre-book alternatives for tight connections, or hold an outbound flight a few minutes. The decision is taken 2 hours before the inbound flight's scheduled departure from its origin (t0). The model estimates P(arrives at DTW 15+ minutes late) at t0.

Every later pipeline choice comes from that:

**1. What is allowed as a feature.** Only information that exists at t0. This rules out the flight's own departure and arrival times and delays, taxi times, and anything measured after t0. It also rules out cancellations so far today: BTS does not record at what time a flight was cancelled, so I cannot prove a cancellation was known before t0. The previous leg of the same aircraft is allowed only if it actually departed before t0.

**2. Target definition.** `ARR_DEL15` (arrived 15+ minutes after schedule), the US industry standard. Cancelled and diverted flights (469 rows, 4.4%) are excluded: they never arrive, so they are a different problem from lateness, and most cancellations are decided well before the 2-hour window.

**3. Split.** Chronological by day and not random.

| Set | Days | Rows | Share late |
| --- | --- | ---: | ---: |
| Train | 1-16 March | 5,133 | 28.1% |
| Validation | 17-21 March | 1,644 | 33.0% |
| Test | 22-31 March | 3,459 | 20.7% |

A random row split would leak. Flights on the same day share weather, congestion and other features. With shuffled rows, every test flight from a bad day would have dozens of training rows from the same storm afternoon, and the score would measure memory of that day instead of performance on an unseen day, which is the only situation the model meets in real use. I placed the boundary so validation holds the storm's tail, so that the threshold is not tuned only on calm days, while training most of the storm.

**4. Metric and threshold.** Accuracy is meaningless here: predicting "on time" for everyone scores 79% on tests and doesn’t protect anyone. Model quality is compared with ROC AUC, PR AUC and log loss. The decision threshold is set by business cost, not by 0.5:

- **False positive** (protection activated, flight on time): $500 per flight. Protection mainly means holding an outbound connection for about 5 minutes. US passenger airlines' direct cost of aircraft block time was $98.41 per minute in 2025 ([A4A](https://airlines.org/dataset/u-s-passenger-carrier-delay-costs/), from DOT Form 41 data), so 5 minutes is about $492.
- **False negative** (late flight, nothing done): $2,000 per flight. A misconnected passenger loses on average 449 minutes ([Barnhart, Fearing & Vaze, MIT](https://dspace.mit.edu/handle/1721.1/89059), model of 2007 US traffic); at the FAA value of passenger time, $47 per hour, that is about $350 per passenger, before the meals and hotels that the largest US airlines commit to for airline-caused disruptions ([DOT dashboard](https://runwaygirlnetwork.com/2023/05/dot-airlines-compensate-stranded-passengers/)). Assumed: about 120 seats, 82.5% load factor (US domestic, March 2026, [FRED](https://fred.stlouisfed.org/series/LOADFACTORD)), half of passengers connecting at DTW, and 10% of those missing their connection when the inbound is 15+ minutes late, giving about 5 misconnects per late flight, or roughly $1,750 + $250 of care = $2,000.

The unit costs are sourced; the passenger numbers per flight are my assumptions and are the weakest part.

The threshold minimising total cost on validation is applied once to test. With a 4:1 cost ratio, decision theory gives an optimal threshold of 500 / (500 + 2,000) = 0.20 for a calibrated model, the validation sweep independently chose 0.20.

## Data preparation & feature engineering

Twelve engineered features describe the three things known at t0: the schedule, the aircraft's previous leg, and how the day is going.

### Time handling

Only the scheduled departure is in UTC. Every other time is derived from durations: scheduled arrival = scheduled departure + scheduled elapsed time. actual departure = scheduled departure + departure delay; actual arrival = scheduled arrival + arrival delay. This avoids two traps: overnight flights whose local arrival time is smaller than their departure time, and comparing clocks across time zones. March 2026 also contains the US switch to summer time (8 March).

### Features

| Feature | Known at t0 because | Why it should predict delay |
| --- | --- | --- |
| Scheduled arrival hour, 5 bins (DTW local) | Schedule | Delays accumulate through the day, later flights inherit earlier problems |
| Scheduled elapsed time | Schedule | Long flights can recover time in the air, short ones cannot |
| Carrier: DL, OO, NK, YX, other | Schedule | Some companies are more reliable than others grouping avoids tiny categories |
| Origin: top 15 by training volume, rest as "other" | Schedule | Airport-specific congestion and weather exposure |
| Turnaround buffer: minutes between the inbound leg's scheduled arrival and our scheduled departure | Both are schedule times | A tight flight change cannot absorb lateness |
| Inbound departure delay, log(1 + minutes), only if the inbound leg left before t0, else 0 | Observed before t0 by construction | Delay propagates from one leg to the next |
| Delay the buffer cannot absorb | Built from the two above | States directly how the propagation happens. |
| Inbound status: departed before t0 / not yet departed / no inbound leg | Observed at t0 | Tells the model whether the delay value is real or a filler 0. A plane that has not left 2 hours before is itself a warning sign |
| Origin day-so-far late rate | Departures before t0 | Weather and congestion at the origin. |
| DTW day-so-far late rate | Arrivals before t0 | Problems at Detroit itself (e.g. the local spike on 13 March). |
| National day-so-far late rate | Departures before t0 | System-wide disruption (15-16 March) |
| DTW arrivals scheduled in the same hour | Schedule, including later-cancelled flights | Runway and gate congestion |

**Inbound leg matching.** Flights are sorted by tail number and scheduled departure. The previous row of the same aircraft counts as the inbound leg only if it lands at our origin and is scheduled to arrive within the previous 12 hours. Otherwise, the status is "no inbound leg".

**Day-so-far rates with shrinkage.** A problem I encountered is that early in the day few flights have departed, so situations such as 1 of 2 late = 50% were creating issues. The solution is that each rate is shrunk toward a prior with 10 imaginary flights: rate = (late + 10 x prior) / (flights + 10). The priors (27.9% for departures, 28.1% for DTW arrivals) are computed from training days only, so validation and test never shape them.

**Log transform.** Inbound delays are mostly 0, but up to 555 minutes. On the raw scale, the 82 training flights with non-zero propagated delay, all 82 of them late, produced a near-separating feature whose weight kept growing and that gradient descent could not converge on. log1p compresses the tail, keeps 0 at 0, and raises the test ROC AUC from 0.768 to 0.779.

**Encoding.** The top-15 origin list, means and standard deviations are all fitted on training rows only.

### Considered and rejected

- **Own departure/arrival delays, taxi times:** known only after t0 so it would cause leakage.
- **Cancellations so far today:** cancellation time is not recorded, so I cannot prove it was known at t0.
- **Day of week / weekend flag:** training has 16 days, so each weekday appears 2-3 times; the training weekends (1, 7, 8, 14, 15 March) include three of the worst storm days, so the model would learn the weather, not the weekday therefore I removed it.
- **Distance:** Basically the same information as scheduled elapsed time, so I chose only one.
- **Origin target encoding:** more information than top-15 one-hot, but I preferred the simpler, explainable option.

## Modeling: three implementations, one model

The target is binary, late or not late. So the model I chose is L2-regularised logistic regression. All three implementations agree with each other on every coefficient and produce identical test metrics.

### Test-set results (22-31 March, 3,459 flights)

| Model | ROC AUC | PR AUC | Log loss | Threshold | Total cost | Cost per flight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline: training late rate for every flight | 0.500 | 0.207 | 0.524 | n/a (never protects) | $1,432,000 | $414 |
| Policy: protect every flight | - | - | - | - | $1,371,500 | $397 |
| scikit-learn `LogisticRegression` | 0.779 | 0.547 | 0.416 | 0.20 | $799,500 | $231 |
| PyTorch manual loop | 0.779 | 0.547 | 0.416 | 0.20 | $799,500 | $231 |
| PyTorch `nn.Module` + `torch.optim` | 0.779 | 0.547 | 0.416 | 0.20 | $799,500 | $231 |

At threshold 0.20 on test: 483 late flights protected, 233 missed, 667 unnecessary protections, 2,076 correctly left alone (precision 42%, recall 67.5%). On validation the same models reach ROC AUC 0.793 and cost $413,000 against $1,086,000 for never protecting.

| Agreement | Max coefficient difference | Max probability difference (test) |
| --- | ---: | ---: |
| sklearn vs PyTorch manual | 4.4 x 10^-6 | 1.4 x 10^-6 |
| sklearn vs PyTorch nn.Module | 4.4 x 10^-6 | 1.4 x 10^-6 |
| PyTorch manual vs nn.Module | 2.2 x 10^-16 | 2.2 x 10^-16 |

### Tuning (sparingly)

| C | 0.001 | 0.01 | 0.1 | 1 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation log loss | 0.5603 | 0.5155 | 0.5006 | 0.4991 | 0.4989 |

C = 10 is nominally best, but by 0.0002, which is noise. The rule I used: among values within 0.001 of the best validation log loss, take the strongest regularisation, which gives C = 1. Learning rate: 1.0, the value in {0.05, 0.1, 0.5, 1.0} with the lowest validation log loss after 5,000 steps among those whose training loss did not diverge (also the largest tried).

### What went wrong first, and how I found it

The first run (raw delay minutes, C = 10) gave a coefficient gap of 0.48 between sklearn and both PyTorch versions, while the two PyTorch versions agreed with each other to 10^-8. That pattern pointed at convergence rather than a bug: both PyTorch runs had travelled the same path and stopped at the same place. The gap sat entirely on the propagated-delay feature: non-zero for only 82 training flights, all of them late. A feature that perfectly separates its subset pushes its weight toward infinity under weak regularisation; sklearn's L-BFGS solver gets far along that direction, plain gradient descent crawls. Two changes fixed it: the log transform (gap down to 0.035) and the stronger-regularisation rule for C (gap down to 4 x 10^-6). Neither change made the model worse: test ROC AUC rose from 0.768 to 0.779.

### What the weights say

The largest weights are the propagated delay (+1.12), inbound status "departed before t0" (-1.22) and "not yet departed" (+0.93) relative to "no inbound leg", the evening arrival bins (+0.9 to +1.0) and the national day-so-far rate (+0.42). The turnaround buffer (+0.02) and DTW arrivals in the same hour (-0.04) contribute almost nothing.

## Limitations & next steps

The model is useful at the estimated costs, but four limitations bound how far the numbers can be trusted.

| Limitation | Evidence | How I would address it |
| --- | --- | --- |
| Business costs are estimates | $500 / $2,000 use sourced unit costs but assumed passenger numbers (seats, connecting share, misconnect rate); the threshold and every cost number depend on their ratio | Get the airline's actual re-accommodation and misconnect costs; the dashboard already lets the costs be changed without retraining |
| Cancelled flights are excluded, and they cluster on bad days | 251 of 444 cancellations fall on 13-18 March, so the dropped rows are the worst cases of the worst days and the evaluation looks easier than reality | Model defined as three outcomes (on time / late / cancelled), or count cancellations as "disrupted" in a second model and then compare |
| Only one month being tested | The model has only seen few days in training and some included a mayor weather disruption so it couldn’t learn weekly effects (if there are actually any) | Train on several months and add the day of week once each weekday appears often enough |
| Two features did not help | Turnaround buffer and DTW same-hour arrivals have near-zero weights. | Drop both features in a simpler model |

Other limitations:

- Inbound legs from the previous evening are not matched for 1 March as we dont have the data for february.
- No weather data. The day-so-far rates act as a substitute, but a proper weather forecast available at t0 would likely be the strongest missing feature.
- Delay propagation is only one leg deep. The inbound leg's own inbound is ignored; a chain feature could capture delays building up across the day.

## Generative AI use disclosure

I used Claude for this assignment for the following areas:

- **Dataset search and framing:** Once I knew a business problem I wanted to do, Claude proposed candidate datasets and some questions the framing had to answer in order to get the ball rolling. After this I fully chose the dataset (BTS flights), the airport (DTW), the month (March 2026 over October 2025), the decision point (2 hours before departure), the 15-minute target, dropping cancelled and diverted flights, and the chronological split with a bad day in validation.
- **Features and fixes:** Once I had a list of features I wanted to include (like the inbound leg delays) Claude challenged some of them and proposed alternatives (like for example the delay the buffer cannot absorb). I decided what to keep. Then, once the code was made and ran Claude helped diagnose the issues in the model and helped fix them. For example, the implementation of the log transform for the Inbound delays was fully Claude’s idea.
- **Code:** `train.py` and `app.py` were written by Claude from those decisions. I chose the regularisation rule (C = 1) after it was explained.
- **Report:** Once the model was finished, Claude helped structure what needed to be included in the report in order to make sure I didn’t forget anything. Also, the formatting and all the tables found in the report were first created by Claude then modified by me. (except the test results, those are 100% Claude)
- **Costs** ($500 / $2,000) were first estimated by Claude; Claude then found the sources used in the framing section, and the passenger-number assumptions are labelled as such.
