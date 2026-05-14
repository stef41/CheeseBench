# CheeseBench: Task Descriptions and Neuroscience Grounding

## Task Overview

CheeseBench comprises nine tasks drawn from classical behavioral neuroscience paradigms used to assess cognitive abilities in rodents. Each task targets a distinct cognitive domain and is implemented as a grid-based environment where the agent must learn from reward feedback.

We adopt the terminology of the rodent behavioral literature. A \textbf{trial} is a single episode: the agent is placed at a start location, takes actions (\textbf{steps}) until it either reaches the goal or exhausts a step budget, and then receives a binary outcome (success/failure). A \textbf{session} groups consecutive trials, mirroring the rodent convention where one session corresponds to one day of testing. The agent's conversation history is maintained across all trials within a task, so the model can learn from prior experience---analogous to a rodent that remembers previous runs through the maze. Table~\ref{tab:tasks} summarizes the tasks, their cognitive domains, and the number of trials available.

| Task | Cognitive Domain | Sessions | Trials/Session | Total Trials | Max Steps/Trial | Grid/Arena Size |
|------|-----------------|----------|---------------|-------------|----------------|----------------|
| Morris Water Maze | Spatial learning & memory | 5 | 4 | 20 | 500 | Circular, $r{=}9$ cells |
| Barnes Maze | Spatial reference memory | 4 | 4 | 16 | 300 | $15{\times}15$ grid |
| T-Maze | Working memory (alternation) | 4 | 10 | 40 | 200 | Stem$=$3, arm$=$2 cells |
| Radial Arm Maze | Working & reference memory | 5 | 4 | 20 | 400 | $25{\times}25$ grid, 8 arms |
| Star Maze | Complex spatial navigation | 8 | 5 | 40 | 300 | $25{\times}25$ grid, 5 arms |
| Operant Chamber | Instrumental conditioning | 5 | 10 | 50 | 100 | Small chamber (phase-based) |
| Shuttle Box | Avoidance learning | 2 | 20 | 40 | 50 | 2-compartment (phase-based) |
| Place Preference | Reward-place association | 6 | 2 | 12 | 300 | 2-chamber continuous |
| DNMS Task | Working memory (non-match) | 25 | 4 | 100 | 50 | Screen-based (phase-based) |

## Detailed Task Descriptions

### 1. Morris Water Maze (MWM)
**Cognitive domain:** Spatial learning and memory (hippocampus-dependent).

The agent is placed at random locations in a circular arena and must navigate to find a hidden platform whose location remains fixed across trials. The platform is invisible; success requires the agent to form a spatial representation using distal cues. Each session comprises 4 trials, with 5 sessions total (20 trials). A reward of +1.0 is given for reaching the platform, while each step incurs a $-$0.01 penalty to encourage efficient navigation.

**Rodent evidence:** The standard MWM acquisition protocol uses semi-random distal start positions (e.g.\ N, E, SE, NW) with 4 trials per day and a 15-s inter-trial interval on the platform \citep{vorhees2006morris}. Each trial has a 2-min time limit for rats (1 min for mice). Vorhees \& Williams report that "5–6 days (20–24 trials) is typically sufficient in a 210~cm maze for rats or in a 122~cm maze for mice to reach asymptotic performance." Rats trained with a standard 10~cm$^2$ platform reach asymptotic escape latency within approximately **20 trials** \citep{vorhees2006morris}. In a smaller 122~cm tank, rats approach asymptotic performance by day 2–3 ($\sim$8--12 trials), though this is considered suboptimal because it allows non-spatial strategies. After acquisition, a 30-s probe trial (platform removed) is given 24~h later to assess reference memory, followed by a 5-day reversal phase (platform relocated to opposite quadrant). The task is highly sensitive to hippocampal lesions and NMDA receptor blockade, making it the gold standard for spatial memory assessment. **CheeseBench provides 20 trials, matching the canonical acquisition protocol.**

### 2. Barnes Maze
**Cognitive domain:** Spatial reference memory (hippocampus-dependent).

The agent is placed at the center of a circular platform with multiple holes around the perimeter. One hole leads to an escape box (the target); all others are blocked. The agent must use spatial cues to identify and navigate to the correct escape hole across sessions. Each session has 4 trials for 4 sessions (16 trials total). A +1.0 reward is given for finding the target hole.

**Rodent evidence:** The Barnes maze was introduced by Barnes (1979) as a less stressful alternative to the MWM, exploiting rodents' natural aversion to open, brightly-lit spaces \citep{barnes1979memory}. The standard protocol described by Patil et al.\ (2009) and Sunyer et al.\ (2007) consists of 1 habituation trial (mouse guided to shelter), followed by a **4-day acquisition period** where the mouse explores freely for 3~min per day, and a probe trial on day~5 \citep{vale2018barnes}. Learning is quantified by the decrease in errors (wrong hole pokes) and latency to reach the shelter across acquisition days. In the defense-escape variant by Vale et al.\ (2018), mice memorize the shelter location after a single 7-min exploration session and escape accurately to it with 97\% accuracy \citep{vale2018barnes}. In the more common multi-trial protocol, mice are given 3--4 trials/day over 4~days (**12--16 total trials**), reaching criterion by day~4, with a long-term memory probe at day~12 \citep{patil2009barnes}. **CheeseBench provides 16 trials, matching the standard 4-day $\times$ 4-trial protocol.**

### 3. T-Maze
**Cognitive domain:** Working memory and spatial alternation.

The agent starts at the base of a T-shaped maze and must choose between the left and right arms. The task implements a rewarded alternation paradigm: the agent must alternate its arm choices across trials to receive rewards. This tests working memory, as the agent must remember its previous choice. Each session has 10 trials for 4 sessions (40 trials total). Correct alternation yields a +1.0 reward; choosing the wrong arm gives a $-$0.5 penalty.

**Rodent evidence:** In the automated T-maze forced alternation task described by Shoji et al.\ (2012), each trial consists of a forced-choice run (animal directed to one arm) followed by a free-choice run (animal must choose the opposite arm for reward) \citep{shoji2012tmaze}. Mice receive 10 consecutive trials per session per day (50-min cutoff). The criterion is a group mean of **80\% correct** responses in a session. Control C57BL/6J mice reach this criterion in approximately **1--2 weeks** of daily testing, corresponding to **70--140 total trials** (10 trials $\times$ 7--14 sessions) \citep{shoji2012tmaze}. In the left-right discrimination variant (reference memory), mice receive 10--20 trials/session and reach 80\% criterion over a similar time course. The task depends critically on hippocampal and prefrontal function. Delayed alternation with 3--60~s delays reveals graded working memory load effects. **CheeseBench provides 40 trials, roughly half the minimum rodent requirement, testing whether VLMs can learn faster than the biological benchmark.**

### 4. Radial Arm Maze (RAM)
**Cognitive domain:** Working memory and spatial reference memory.

The agent is placed at the center of an 8-arm maze. Some arms are baited (contain rewards) and others are never baited. The agent must learn which arms contain rewards (reference memory) while avoiding re-entering arms already visited within a trial (working memory). Each session has 4 trials for 5 sessions (20 trials total). Collecting a reward from a baited arm yields +0.25, with a +1.0 bonus upon collecting all rewards.

**Rodent evidence:** The radial arm maze was introduced by Olton, Collison \& Werz (1977) to assess spatial memory in rats \citep{olton1977radial}. In the eight-arm radial water maze variant described by Penley et al.\ (2013), rats receive **4 trials per day** with a 90-s inter-trial interval and a 120-s maximum trial duration \citep{penley2013radial}. On each day, 4 arms contain escape platforms; one platform is removed after each successful trial, progressively increasing working memory demand across trials~1--4. Significant effects of neonatal brain injury on both working and reference memory errors were found by **day~11** of testing ($\sim$44 trials) \citep{penley2013radial}. Even with observed effects, **15--20 days of testing** ($\sim$60--80 trials) are recommended to reduce variability. In the original land-based RAM with food reward, Olton (1977) reported that rats learned to enter each of 8 arms without repetition within **4--8 testing days** \citep{olton1977radial}. **CheeseBench provides 20 trials, which is below the ~44-trial minimum for significant effects, deliberately creating a challenging sample-efficiency test.**

### 5. Star Maze
**Cognitive domain:** Complex spatial navigation and strategy selection.

The agent navigates a pentagram-shaped maze to find a fixed destination, testing both egocentric (body-turn sequence) and allocentric (landmark-based) navigation strategies. The maze consists of five central alleys forming a pentagon with five radiating arms. Each session has 5 trials for 8 sessions (40 trials total). Reaching the target yields +1.0.

**Rodent evidence:** The star maze was developed by Rondi-Reig and colleagues to dissociate sequential egocentric navigation (requiring hippocampus + dorsomedial striatum) from allocentric spatial strategies (hippocampus-dependent) \citep{rondi2006starmaze, fouquet2013starmaze}. In the original rodent protocol, Rondi-Reig et al.\ (2006) tested mice over **3~days $\times$ 6~trials = 18~trials**, finding that control mice reliably learn the task while L7-PKCI transgenic mice with cerebellar LTD deficits fail to adopt sequential egocentric strategies \citep{rondi2006starmaze}. Fouquet et al.\ (2013) used a massed 1-day protocol of **10 sessions $\times$ 4 trials = 40 trials** (10-min ITI, 40-min inter-session interval), with a criterion of $\leq$1 error in the last 4 trials. They demonstrated that hippocampal lesions impair both strategies (mice adopt a serial strategy instead) while dorsomedial striatal lesions severely impair all goal-directed navigation \citep{fouquet2013starmaze}. Iglói et al.\ (2009) extended the paradigm to human fMRI studies, confirming the dual-strategy framework. In virtual star maze studies with older human adults, participants are provided 9 learning trials with a 90-s time limit, reaching $\sim$55--77\% success by trial~9 \citep{zhang2021starmaze}. **CheeseBench provides 40 trials, matching the 18--40 trials used in rodent protocols, to allow exploration of both navigation strategies.**

### 6. Operant Chamber (Skinner Box)
**Cognitive domain:** Instrumental conditioning and reinforcement learning.

The agent is placed in a small chamber and must learn to perform a specific action (press a lever/nose-poke) to obtain a reward. The task tests the agent's ability to learn stimulus–response–outcome contingencies. Each session has 10 trials for 5 sessions (50 trials total). Correct responses yield +1.

**Rodent evidence:** Operant conditioning, formalized by Skinner (1938), is the fundamental paradigm for studying instrumental learning. Acquisition proceeds in stages: (1) magazine training (1--2 sessions of 30~min to learn food-cup approach), (2) continuous reinforcement on FR1 schedule (every lever press rewarded), (3) transition to leaner schedules (FR5, VR, etc.). On the FR1 schedule, naive rats typically achieve stable lever-pressing within **3--5 daily sessions** (30--60~min each), accumulating **100--300 reinforced responses** across these sessions \citep{skinner1938behavior}. Martin \& Iceberg (2015) report that in a social motivation operant paradigm, mice require daily 30-min shaping sessions until achieving 10 lever presses in 3/5 sessions, with most mice reaching asymptotic performance within **7 days** and completing 20 test sessions thereafter \citep{martin2015operant}. Progressive ratio breakpoints (where the animal ceases responding as the ratio increases) typically stabilize within **5--10 sessions** \citep{hodos1961progressive}. **CheeseBench provides 50 trials, a sufficient budget given that the core operant association (action $\rightarrow$ reward) is learned within 3--7 sessions in rodents.**

### 7. Shuttle Box
**Cognitive domain:** Avoidance learning (classical + instrumental conditioning).

The agent is in a two-compartment box. A conditioned stimulus (CS) is presented, followed by an aversive unconditioned stimulus (US). The agent must learn to shuttle from one compartment to the other to avoid the US. This involves learning the CS–US association (classical conditioning) and the avoidance response (instrumental conditioning). Each session has 20 trials for 2 sessions (40 trials total). Successful avoidance (shuttling during cue) yields +1.0, escaping after shock onset yields +0.5, and failing to escape yields $-$0.5.

**Rodent evidence:** Two-way active avoidance in the shuttle box is one of the oldest paradigms in learning research, originating with Mowrer \& Lamoreaux (1946) \citep{mowrer1946shuttle}. In the standard protocol, a CS (tone or light, 5--10~s duration) precedes a US (footshock, 0.3--0.5~mA) in one compartment; the animal must cross to the other compartment to avoid the US. Happel et al.\ (2015) demonstrated that gerbils learn discrimination between two ICMS sites within 3 training sessions using 30--90 trials/session ($\sim$90--270 total trials), with a d$'$~$>$~1 criterion \citep{happel2015shuttlebox}. In standard rat protocols, avoidance rates typically rise from $\sim$0\% to $\geq$70--80\% within **2--5 sessions** of **30 trials** each (**60--150 total trials**) \citep{mowrer1946shuttle}. Escape latencies below 2~s indicate stable shock-control responding. Notably, avoidance learning shows considerable individual variation: some animals ("good avoiders") reach criterion within 1--2 sessions while others ("poor avoiders") require 5+ sessions. **CheeseBench provides 40 trials, which falls at the lower end of the rodent requirement, testing whether VLMs can acquire avoidance associations rapidly.**

### 8. Place Preference (Conditioned Place Preference, CPP)
**Cognitive domain:** Reward-context association (Pavlovian conditioning).

The agent alternates between two visually distinct compartments. One compartment is paired with a reward during conditioning; the other is not. After conditioning, the agent's preference for the reward-paired compartment is measured. Each session has 2 trials for 6 sessions (12 trials total). A small continuous reward of +0.1 per step is given while the agent occupies the paired compartment during conditioning.

**Rodent evidence:** CPP is a standard Pavlovian paradigm for measuring the incentive properties of rewards by associating them with distinct environmental contexts \citep{tzschentke2007cpp}. The protocol consists of three phases: (1)~pre-conditioning (2--3 days of 15-min free exploration to establish baseline preference), (2)~conditioning (alternating confinement in reward-paired vs.\ neutral compartment), and (3)~post-conditioning test (free exploration). Blanco-Gandía et al.\ (2018) used **4 conditioning days** with 2 pairings/day (30~min each), for a total of **4 reward--context pairings**. The total protocol spans $\sim$8~days \citep{blanco2018cpp}. CPP is reliably established with as few as **2--4 conditioning sessions** for pharmacological rewards and **3--6 sessions** for natural rewards (food, social interaction) \citep{tzschentke2007cpp}. The critical measure is a statistically significant increase in time spent in the reward-paired compartment relative to pre-conditioning baseline. Weekly extinction sessions (unpaired exposure) can reverse the preference, and reinstatement can be triggered by reward priming. **CheeseBench provides 12 trials over 6 sessions, matching the standard 4--6 conditioning pairings required in rodent protocols.**

### 9. Delayed Non-Match to Sample (DNMS) Task
**Cognitive domain:** Working memory and pattern separation.

The agent is presented with a sample stimulus at a location, then after a delay must select the *non-matching* (novel) location. This tests the ability to hold a spatial location in working memory and discriminate between similar stimuli. Each session has 32 trials for 25 sessions (800 trials total). Correct non-match responses yield +1.

**Rodent evidence:** In the touchscreen Trial-Unique delayed Nonmatching-to-Location (TUNL) task described by Oomen et al.\ (2013), rats are trained through multiple pretraining stages (screen touch, match-to-position, learning the non-match rule) spanning **10--30 sessions** before the main task begins \citep{oomen2013tunl}. In the main task, rats require an average of **1,939~$\pm$~79 trials** over **34~$\pm$~1 daily sessions** to reach a criterion of **80\% correct on 2 consecutive days** at the largest spatial separations \citep{oomen2013tunl}. The LD (large distance) variant takes 20--25 sessions for rats and 20--40 sessions for mice. When smaller separations are introduced (requiring finer pattern separation), performance drops significantly, demonstrating the task's sensitivity to dentate gyrus function. The high trial count reflects three layered cognitive demands: (1)~learning the non-match rule, (2)~maintaining sample location across a delay, and (3)~discriminating between spatially similar locations while suppressing proactive interference from previous trials. **CheeseBench provides 800 trials, approximately 41\% of what rats require, making this the most challenging task in the benchmark relative to rodent performance.**

## Rodent Learning Efficiency: Summary and Comparison

| Task | Rodent Trials to Criterion | Source (Key Finding) | CheeseBench Trials | Ratio (Bench/Rodent) |
|------|---------------------------|---------------------|-------------------|---------------------|
| Morris Water Maze | **20--24** (5--6 days $\times$ 4 trials) | Vorhees \& Williams 2006: asymptote within 20 trials | 20 | 0.83--1.0$\times$ |
| Barnes Maze | **12--16** (4 days $\times$ 3--4 trials) | Barnes 1979; Patil et al.\ 2009: criterion by day 4 | 16 | 1.0--1.3$\times$ |
| T-Maze | **70--140** (1--2 weeks $\times$ 10 trials/day) | Shoji et al.\ 2012: 80\% correct in 7--14 sessions | 40 | 0.29--0.57$\times$ |
| Radial Arm Maze | **44--80** (11--20 days $\times$ 4 trials) | Penley et al.\ 2013: significant effects by day 11 | 20 | 0.25--0.45$\times$ |
| Star Maze | **18--40** (1--3 days $\times$ 4--6 trials or 10 sessions $\times$ 4 trials) | Rondi-Reig et al.\ 2006; Fouquet et al.\ 2013 | 40 | 1.0--2.2$\times$ |
| Operant Chamber | **100--300 responses** (3--7 sessions) | Skinner 1938; Martin \& Iceberg 2015: asymptote in 7 days | 50 | variable |
| Shuttle Box | **60--150** (2--5 sessions $\times$ 30 trials) | Mowrer \& Lamoreaux 1946; Happel et al.\ 2015 | 40 | 0.27--0.67$\times$ |
| Place Preference | **4--6 pairings** ($\sim$8 days total) | Blanco-Gandía et al.\ 2018; Tzschentke 2007 | 12 | 2.0--3.0$\times$ |
| DNMS Task | **$\sim$1,939** ($\sim$34 sessions) | Oomen et al.\ 2013: 80\% correct at largest separation | 100 | 0.05$\times$ |

CheeseBench deliberately calibrates its trial budget against the rodent literature. For tasks where rodent learning is rapid (MWM, Barnes Maze, CPP), CheeseBench provides a trial budget at or above the rodent requirement. For tasks where rodents need many trials (T-Maze, RAM, Shuttle Box, DNMS), CheeseBench provides fewer trials than rodents require, creating a stringent test of sample efficiency. This design makes it possible to quantify how VLMs compare to biological learners: an agent that matches rodent-level criterion performance within the allotted trials would demonstrate remarkable few-shot learning, while agents that fail to reach criterion reveal the gap between current AI and biological intelligence.

The three orders of magnitude separating the easiest tasks ($\sim$12--20 trials for spatial reference memory) from the hardest ($\sim$1,939 trials for DNMS) reflect a fundamental organizing principle in neuroscience: simple stimulus-response associations are learned rapidly, while tasks requiring rule abstraction, working memory maintenance, and interference suppression demand extensive experience.

## References

- \bibitem{defiebre2006spatial} de Fiebre, N.C., Bhatt, R.S. \& de Fiebre, C.M. (2006). Spatial learning and psychomotor performance of C57BL/6 mice: age sensitivity and reliability of individual differences. *Age*, 28(3), 235--253. PMC3259155.

- \bibitem{barnes1979memory} Barnes, C.A. (1979). Memory deficits associated with senescence: a neurophysiological and behavioral study in the rat. *Journal of Comparative and Physiological Psychology*, 93(1), 74--104.

- \bibitem{harrison2006barnes} Harrison, F.E., Hosseini, A.H. \& McDonald, M.P. (2009). Endogenous anxiety and stress responses in water maze and Barnes maze spatial memory tasks. *Behavioural Brain Research*, 198(1), 247--251. PMC1783636.

- \bibitem{patil2009barnes} Patil, S.S., Sunyer, B., Höger, H. \& Lubec, G. (2009). Evaluation of spatial memory of C57BL/6J and CD1 mice in the Barnes maze, the Multiple T-maze and in the Morris water maze. *Behavioural Brain Research*, 198(1), 58--68.

- \bibitem{shoji2012tmaze} Shoji, H., Hagihara, H., Takao, K., Hattori, S. \& Miyakawa, T. (2012). T-maze Forced Alternation and Left-right Discrimination Tasks for Assessing Working and Reference Memory in Mice. *J. Vis. Exp.*, (60), e3300. PMC3399492.

- \bibitem{olton1977radial} Olton, D., Collison, C. \& Werz, M. (1977). Spatial memory and radial-arm maze performance of rats. *Learning and Motivation*, 8, 289--314.

- \bibitem{penley2013radial} Penley, S.C., Gaudet, C.M. \& Threlkeld, S.W. (2013). Use of an Eight-arm Radial Water Maze to Assess Working and Reference Memory Following Neonatal Brain Injury. *J. Vis. Exp.*, (82), e50940. PMC4030456.

- \bibitem{rondi2006starmaze} Rondi-Reig, L., Petit, G.H., Tobin, C., Tonegawa, S., Mariani, J. \& Bhatt, D.H. (2006). Impaired sequential egocentric and allocentric memories in forebrain-specific--NMDA receptor knock-out mice during a new task dissociating strategies of navigation. *J. Neurosci.*, 26(15), 4071--4081. PMC6673881.

- \bibitem{fouquet2013starmaze} Fouquet, C., Babayan, B.M., Watilliaux, A., Bontempi, B., Tobin, C. \& Rondi-Reig, L. (2013). Complementary roles of the hippocampus and the dorsomedial striatum during spatial and sequence-based navigation behavior. *PLoS One*, 8(6), e67232. PMC3695082.

- \bibitem{zhang2021starmaze} Zhang, J.X., et al. (2021). Age-related impairment of navigation and strategy in virtual star maze. *BMC Geriatrics*, 21(1), 108. PMC7866711.

- \bibitem{skinner1938behavior} Skinner, B.F. (1938). *The Behavior of Organisms: An Experimental Analysis*. Appleton-Century-Crofts, New York.

- \bibitem{martin2015operant} Martin, L. \& Iceberg, E. (2015). Quantifying Social Motivation in Mice Using Operant Conditioning. *J. Vis. Exp.*, (102), e53009. PMC4598097.

- \bibitem{hodos1961progressive} Hodos, W. (1961). Progressive ratio as a measure of reward strength. *Science*, 134(3483), 943--944.

- \bibitem{mowrer1946shuttle} Mowrer, O.H. \& Lamoreaux, R.R. (1946). Fear as an intervening variable in avoidance conditioning. *Journal of Comparative Psychology*, 39(1), 29--50.

- \bibitem{happel2015shuttlebox} Happel, M.F.K., Deliano, M. \& Ohl, F.W. (2015). Combined Shuttle-Box Training with Electrophysiological Cortex Recording and Stimulation. *J. Vis. Exp.*, (104), e53004. PMC4692667.

- \bibitem{tzschentke2007cpp} Tzschentke, T.M. (2007). Measuring reward with the conditioned place preference (CPP) paradigm: update of the last decade. *Addiction Biology*, 12(3--4), 227--462.

- \bibitem{blanco2018cpp} Blanco-Gandía, M.C., et al. (2018). Reinstatement of Drug-seeking in Mice Using the Conditioned Place Preference Paradigm. *J. Vis. Exp.*, (136), e56983. PMC6101638.

- \bibitem{oomen2013tunl} Oomen, C.A., et al. (2013). The touchscreen operant platform for testing working memory and pattern separation in rats and mice. *Nature Protocols*, 8(10), 2006--2021. PMC3982138.
