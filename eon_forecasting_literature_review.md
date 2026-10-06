# Literature review for E.ON day ahead household energy forecasting

Research checked on 6 October 2026. Scope: the supplied Bringing the Heat challenge, comprising 410 households with heat pumps, optional PV ownership metadata, 15 minute consumption measurements, and eight hourly weather stations.

## Main conclusion

The literature supports a competition between a few complementary models, followed by calibrated portfolio forecasts and a purchasing rule. It does not establish an absolute winner for these 410 households. No cited study evaluates every candidate on this exact dataset, at the same forecast cutoff, with the same weather availability and purchasing costs.

My first implementation would be a **global LightGBM or CatBoost model with direct forecasts and quantile outputs**, compared with a **global encoder decoder Transformer** and **Chronos-2**. TimesFM-3 is an additional current foundation model challenger where its use terms fit the project. A seasonal ensemble and a pooled historical analogue method provide essential baselines. These are recommendations inferred from the evidence below, not measured results on the hackathon data.

For the complete problem, the architecture matters as much as the forecasting model: preserve the information available at the purchasing cutoff; model changing household behavior; evaluate aggregate demand; calibrate uncertainty using recent observed errors; and choose the procurement quantile using stated costs.

| Part of the problem | First choice for this task | Stronger or more specialized challenger | What would determine the winner |
|---|---|---|---|
| Missing data and alignment | Explicit observation masks, native missing feature handling, causal seasonal filling where needed | Seasonal state space filtering; SAITS for extensive gaps within observed histories | Downstream forecast quality on realistically masked histories |
| Household point forecasting | Global direct LightGBM or CatBoost | Global encoder decoder Transformer; N-HITS | Honest rolling origin error and portfolio purchasing cost |
| Heat pump demand | Weather aware boosted trees and Transformer, with thermal memory features | A physical thermal model with learned residuals when indoor and control measurements exist | Cold weather and post-optimization performance |
| Short household histories | Pooled model with metadata and recent scale features | Historical analogues; Chronos-2 | Separate tests for households with little history |
| Foundation models | Chronos-2 as a practical covariate aware reference | TimesFM-3, TiRex-2, Toto 2.0 | Same inputs, calibration, computation budget, and valid pretraining provenance |
| Household grouping | Known PV flag and building features inside a global model | Feature clustering, limited DTW clustering, or shared model with group effects | Improvement over an ungrouped global model |
| Unknown PV ownership | Calibrated supervised classifier on known labels | Interpretable feature clustering and cautious abstention | Held out household classification and forecast benefit |
| Probabilistic forecasts | Quantile models with recent calibration | Quantile GAM, quantile forest, historical analogues, distributional neural model | Pinball loss or CRPS together with coverage and width |
| Aggregate uncertainty | Direct portfolio distribution or dependence preserving residual scenarios | Empirical copula or conditional flow; coherent probabilistic reconciliation | Portfolio calibration and purchasing risk |
| Procurement | Cost optimal conditional quantile | Scenario optimization with risk and market constraints | Realized cost under identical price assumptions |
| Evaluation | Shared temporal origins and a final untouched period | Repeated seasonal backtests and block based uncertainty estimates | Reproducible gains across seasons and important subgroups |

## How the evidence was assessed

This is a targeted critical literature review, not an exhaustive systematic review with a registered search protocol. Original papers, proceedings, author manuscripts, official model cards, and official documentation were prioritized. Broad forecasting benchmarks were used to identify candidates; energy and household studies were used to judge relevance. Recent preprints and developer benchmark claims are identified as such. Search snippets were used for discovery, with full text checked for the most consequential household and heat pump findings.

Evidence is stronger when it uses multiple datasets, temporally ordered evaluation, appropriate baselines, available code, and inputs that would exist at prediction time. Results based on aggregated loads, perfect future weather, or different forecast horizons require qualification before transferring them to this challenge. Reported percentage gains below belong to the original studies and are not expected hackathon gains.

## The most informative empirical findings

**Heat pump specific evidence.** Semmelmann et al. (2024) studied hourly aggregate demand from 21 German households with water-to-water heat pumps. Transformer methods led the heat pump and combined-load comparisons; separating heat pump and household forecasts gave little benefit among the strongest methods. Crucially, the study uses perfect foresight weather and a small, specific population. Its conclusion motivates a Transformer challenger and a decomposition experiment, not a universal guarantee. [Paper](https://publikationen.bibliothek.kit.edu/1000170854/152908535), [authors' implementation](https://github.com/leloq/load-forecasting-with-heatpumps).

**Individual household probabilistic evidence.** Botman et al. (2025) compare pooled normalized historical profiles with several probabilistic alternatives. Their method needs seven days of target-household history. In Table 1 it has the lowest CRPS on CER and Low Carbon London, but QRA wins on the Belgian Fluvius data, including the 15 minute case. Improvements over the study's baseline are 31.82%, 27.23%, and 29.86% for the pooled method respectively; its last result is not the best on that dataset. The particularly relevant lesson is that simple pooling and forecast combination deserve serious testing. [Paper](https://ftp.esat.kuleuven.be/pub/pub/stadius/ida/reports/24-178.pdf).

**A newer electrical load benchmark.** Hertel et al. (2026), an arXiv preprint, report Transformer advantages across three grid levels. However, their task predicts 96 hourly values with hourly forecast origins; neural models are global whereas LightGBM is local, and weather is reanalysis on two datasets. This does not settle global boosted trees versus Transformers for 96 quarter-hours at one daily cutoff. [Paper](https://arxiv.org/abs/2607.15705).

**Evidence against automatic preference for deep learning.** In the official BuildingsBench results, a persistence ensemble is competitive for residential buildings. In its transfer-learning table, residential NRMSE is 77.20 for a pretrained and fine-tuned Gaussian Transformer, 78.54 for the persistence ensemble, and 80.07 for LightGBM. These results have a different setup and population, but make a strong case for retaining serious baselines. [Benchmark paper](https://arxiv.org/abs/2307.00142), [official results](https://github.com/NatLabRockies/BuildingsBench).

**Energy forecasting competition evidence.** Gaillard, Goude, and Nedellec's GEFCom2014 winning load approach combined temperature scenarios with quantile generalized additive models. Ben Taieb et al. separately demonstrate boosted additive quantile regression for individual smart meters. These papers support nonlinear temperature effects and conditional uncertainty, rather than the assumption that the largest neural network must be best. [GEFCom method](https://www.sciencedirect.com/science/article/abs/pii/S0169207015001545), [smart meter quantile study](https://souhaib-bentaieb.com/publications/smart-meter-quantiles/).

## Define the target and information set first

The challenge provides Total, HeatPump, and Other interval energy. It does not provide PV production or returned energy. Confirm whether Total is gross electricity use, grid import, or another metering boundary, and whether Total equals HeatPump plus Other after accounting for missing or differently measured channels. This determines whether summing households is an appropriate proxy for procurement demand. A column name alone cannot answer that question.

Use a forecast origin o, a target interval t, and a precisely defined information set available by o. The goal is to predict Y(i,t) for household i and the next delivery day. Calendar features for t are available; consumption after o is unavailable; weather forecasts issued by o are available if archived. Include realistic meter publication delays.

If the operational cutoff is before midnight on the preceding day, the delivery-day horizon is longer than the next 24 hours. A same-time-yesterday feature may refer to consumption after the cutoff for later target intervals. Build every lag relative to the actual origin, or explicitly test its availability. A fixed midnight forecast is a useful baseline experiment only if labeled as that experiment.

UTC avoids ambiguous timestamps, while local time is needed to represent human routines. A UTC day has 96 quarter-hours. A local delivery day can have 92 or 100 on daylight-saving transitions, if the applicable timezone observes those changes. Do not remove or duplicate energy to force every local day into 96 bins. Confirm the household geography, delivery timezone, and timestamp interval convention with mentors.

The years covered include possible pandemic and behavior changes as well as heat pump optimization visits. Use only visit information known at the origin. Do not interpret before-versus-after forecast differences as causal energy savings without a separate causal identification design.

## Data preparation and weather

### Missing observations and outliers

Start with a household-by-time completeness audit: duplicate timestamps, inconsistent interval lengths, absent rows, missing numeric fields, channel identities, nonphysical readings, long outages, and dates with changing cohort coverage. Zero consumption and an absent observation are different states. Do not turn missing targets into zero or score imputed targets as actual consumption.

For boosted trees, retain missing covariates and add observation-count or missingness features. For neural models, use supported masks and a documented filling method. A seasonal median or a state space filter is a sensible first fill for short gaps. A Kalman filter uses observations available up to the origin; a smoother must be restricted to that observed context. Interpolation between two past observations is allowed when both were available by the origin. Interpolation across the forecast boundary is not.

SAITS is a strong specialized multivariate imputation candidate in its own literature, but it is not automatically the best preprocessing for forecasting. Test artificial gaps resembling actual missingness and compare downstream forecast error. A method that makes reconstructed histories look smooth can remove important peaks. [SAITS paper](https://www.sciencedirect.com/science/article/pii/S0957417423001203).

Do not forward-fill interval kWh as though it were a continuous meter reading. Do not indiscriminately clip high demand: cold-weather peaks may be exactly what procurement needs to predict. Fit all learned cleaning thresholds, scalers, household profiles, and feature selectors using the training information set.

### Resolution alignment

Keep the 15 minute target if the submitted prediction requires it. Join hourly weather by Weather_ID and valid time, interpreting whether each field is an instantaneous reading, hourly mean, or accumulation. A repeated hourly temperature or interpolation between already issued forecast values can be valid. Upsampling does not create new weather information. Accumulated precipitation and irradiance-related quantities need unit-aware treatment.

As an ablation, sum four interval-energy values to hourly energy, forecast the 24 hourly totals, and disaggregate using quarter-hour shares estimated from past data. Check both hourly and quarter-hour results: smoothing may improve aggregate error while weakening peak timing. If the input were power in kW, hourly means would be appropriate; the supplied target is energy in kWh, so aggregation uses sums.

### What weather model is appropriate

Prefer archived, issued numerical weather forecasts rather than training a weather foundation model for eight stations during the hackathon. Compare available regional forecasts and ensemble products based on local temperature and irradiance skill. ECMWF operates both physics-based IFS and AIFS ensembles; that makes them candidates, not proven winners for these household locations. [Official AIFS ensemble announcement](https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwfs-ensemble-ai-forecasts-become-operational).

Run three clearly distinguished input conditions: past consumption and calendar only; a deployable weather condition using archived forecasts or an explicitly constructed past-only weather predictor; and an optional perfect-weather diagnostic. Observed future weather can quantify an upper-information scenario, but its result is not operational forecast accuracy. A retrospective weather hindcast also needs labeling as a reconstructed experiment rather than a forecast actually available then.

## Level 0 household forecasting models

### Global direct boosted trees

Train one model across households, with rows containing the origin, household, and target interval or lead time. Direct prediction means forecasting each target using the same available history; do not substitute realized earlier target-day consumption into later predictions. Start with a shared model including horizon and clock-time features, and compare it with a few separate horizon groups if validation supports them.

LightGBM is designed for efficient boosting on large tabular data. CatBoost has a particularly useful categorical-feature treatment. Neither original paper proves superiority for this energy dataset. The engineering inference is that these models match the available heterogeneous features and permit rapid feature ablations. [LightGBM](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), [CatBoost](https://proceedings.neurips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html).

Candidate features include known recent load histories; corresponding past weekday slots; recent daily energy and peak statistics; quarter-hour, weekday, holiday, and season; building size and heat pump type; known PV status with an unknown category; weather-station identity; forecast temperature and its recent change; lagged or smoothed temperature representing building thermal memory; and visit status if known. All aggregates must exclude unavailable observations. Missing survey rows should remain represented rather than silently dropping 17 households.

For the conditional mean, use a loss appropriate to the original kWh target, such as squared error. For the median or purchasing quantiles, use quantile loss. Back-transforming a log-scale prediction does not automatically give the conditional mean in kWh. Normalize household scale using past-only statistics if needed; retain scale information so large and small users remain distinguishable.

Global models share information across series and need not assume that every household is identical. Montero-Manso and Hyndman provide a theoretical and empirical rationale for this pooling. A long-history household can still benefit from a local residual correction, but that is a validation question. [Global versus local methods](https://arxiv.org/abs/2008.00444).

### Trained neural candidates

**Encoder decoder Transformer:** my first trained neural challenger for accuracy. Encode observed load and weather history; decode the target day's known calendar and issued weather inputs. Share weights across households and add suitable household descriptors. Consider separate mean and quantile outputs. The current electricity-specific evidence supports trying this relatively plain architecture before elaborate hybrid networks.

**Temporal Fusion Transformer:** attractive when the model must integrate static household metadata, historical measurements, and known future inputs, while exposing variable-selection diagnostics. It was designed for this mix of inputs. Its attention or variable importance is not a causal explanation, and its original results do not prove a win over current models here. [TFT](https://arxiv.org/abs/1912.09363).

**N-HITS:** a valuable architecture challenger based on interpolation across different temporal scales. It can be an economical way to model smooth day profiles, but its long-horizon benchmark improvements should not be imported as expected day-ahead household gains. Make sure the chosen implementation includes the exogenous inputs needed for a fair comparison. [N-HITS](https://arxiv.org/abs/2201.12886).

**PatchTST and TimeXer:** PatchTST makes long contexts more tractable using patches and shared channel processing; TimeXer explicitly addresses exogenous variables. They deserve consideration when history length or covariate integration is limiting. Plain PatchTST's channel independence is not the same as learning all household and weather interactions. Audit how an implementation handles future-known covariates instead of assuming every Transformer supports them. [PatchTST](https://arxiv.org/abs/2211.14730), [TimeXer](https://arxiv.org/abs/2402.19072).

**DeepAR:** a useful established global probabilistic baseline that generates autoregressive future paths. Its chosen observation distribution and sequential sampling can be disadvantages compared with direct quantile forecasting, particularly with near-zero demand or long target horizons. It remains informative as a comparison, rather than an assumed leader. [DeepAR](https://arxiv.org/abs/1704.04110).

Try roughly one to several weeks of input context, then select the length using validation; these are proposed search ranges, not proven optimal hyperparameters. Regularize, use early stopping on later origins, and train several seeds only when differences matter. Millions of overlapping quarter-hour rows are not millions of independent weather or behavior examples.

### Foundation models current to October 2026

| Model | Why it belongs in the shortlist | Material limitation |
|---|---|---|
| Chronos-2 | Native multivariate and covariate-aware forecasts, cross-series learning, direct quantile output; practical first reference | Calibration and household-scale performance still need testing; public pretraining provenance matters |
| TimesFM-3 | Google reports leading general benchmark performance and native multivariate and covariate support | Downloaded weights have noncommercial and nonproduction restrictions; benchmark results are developer reported |
| TiRex-2 | Recurrent architecture with past and future-known covariates and comparatively small active model | New 2026 evidence; distinguish the public release from optimized streaming capabilities described for its Pro offering |
| Toto 2.0 | Broad probabilistic benchmark strength and several model sizes | Benchmark and scaling results do not establish a weather-aware household procurement win; verify the required input interface |

Chronos-2's paper reports strong zero-shot results on fev-bench, GIFT-Eval, and Chronos Benchmark II at release. Its official model card lists 120M parameters and Apache-2.0 licensing. It is my practical first foundation-model experiment, because the information structure fits this task. That preference is an implementation judgment rather than a claim that it still leads every 2026 benchmark. [Paper](https://arxiv.org/abs/2510.15821), [model card](https://huggingface.co/amazon/chronos-2).

TimesFM-3 was announced on 31 August 2026. Google reports leading performance across major general benchmarks and native support for multiple series and covariates. Its downloaded weights are separately restricted to noncommercial, nonproduction use; authorized Google Cloud deployment has different terms. This is a concrete deployment distinction for an E.ON-facing project. [Google announcement](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/), [official usage and license notice](https://github.com/google-research/timesfm).

TiRex-2 and Toto 2.0 are important additions to a current research shortlist. The former emphasizes a recurrent multivariate design; the latter investigates performance scaling across five sizes. Their new results should be treated as model-development evidence, with equal-input tests here deciding practical utility. [TiRex-2 paper](https://arxiv.org/abs/2607.01204), [TiRex-2 model card](https://huggingface.co/NX-AI/TiRex-2), [Toto 2.0 technical report](https://arxiv.org/abs/2605.20119).

Foundation-model rankings depend on metric, task, context length, covariates, training overlap, and whether the model is frozen or fine-tuned. fev-bench explicitly addresses covariates and statistical comparisons; GIFT-Eval covers diverse forecasting conditions; TIME is designed around fresh tasks and contamination concerns. Use these to identify candidates, not to declare the winner for household procurement. [fev-bench](https://arxiv.org/abs/2509.26468), [GIFT-Eval](https://arxiv.org/abs/2410.10393), [TIME](https://arxiv.org/abs/2602.12147).

As additional domain evidence, a September 2026 study reports strong peak-period results for Chronos-2 across several distribution-grid aggregation levels. It is an aggregation and peak study rather than the present household challenge. Another September study of covariate-informed utility forecasts uses daily series and much longer horizons, so it cannot establish next-day quarter-hour performance. [Peak-aware study](https://arxiv.org/abs/2609.18588), [utility covariate study](https://arxiv.org/abs/2609.06656).

A 2026 pretrained model backtested on 2019–2024 data is a retrospective evaluation of today's model, not a system that could actually have been deployed in 2019. Check training-set overlap, obtain decontaminated checkpoints where applicable, and state the external-pretraining assumption. All series supplied for cross-learning must also stop at the common origin; another household's future cannot be smuggled in through batching.

### Essential statistical and analogue baselines

Include same-slot previous-week demand, the last available comparable weekday, and a recent same-slot seasonal average or median. A blend of several such forecasts is stronger than a deliberately weak single baseline. Fit weights only on earlier validation predictions.

A GAM or quantile GAM can express nonlinear temperature effects, calendar smooths, and household variation with comparatively clear behavior. A ridge autoregression is another useful fast reference. SARIMAX and exponential smoothing are worth testing on smoother aggregates; Prophet and a generic LSTM are comparison options, but the selected literature does not justify making either the default winner.

For pooled analogues, retrieve similar normalized past histories across households and form empirical predictive quantiles. Restrict every retrieved window and any associated target to observations available by the origin. This is particularly useful for a household with a short history or changing behavior. It should be compared with a weather-aware pooled model because all households here have heat pumps.

## Heat pump structure and physical modeling

Compare direct Total forecasts with forecasts of HeatPump and Other whose means sum to Total. Even if the measured channels add perfectly, the split model can introduce extra estimation error. If outputs are distributions, summing their equal-level quantiles is not a valid total distribution without assumptions about dependence.

Useful thermal features include nonlinear heating-degree terms, outdoor-temperature changes, past temperature averages, season, heat pump type, and recent heat pump duty behavior. Forecast hot-water activity as well as space heating; do not assume that summer heat pump load is identically zero. If there is a large genuine point mass at zero, a proposed hurdle model separates the probability of operation from the positive consumption distribution. It is an optional ablation, not an established best result for this dataset.

A thermal resistance-capacitance model describes heat exchange and building thermal storage. A hybrid model can combine that structure with a learned forecast residual. NIST's transactive-energy forecasting tool uses a lumped-capacitance formulation for next-day heat pump use and indoor temperature. Such a model becomes much more defensible when indoor temperature, thermostat schedules, supply temperatures, operational mode, and equipment efficiency are measured. [NIST study](https://www.nist.gov/publications/load-forecasting-tool-nist-transactive-energy-market).

The supplied metadata and electricity measurements do not establish that these physical states or controls are observed. Therefore a detailed physics-informed neural network is a higher-risk first choice: its latent parameters may not be identifiable, and an elegant thermal fit does not establish better purchasing forecasts. Begin with weather-aware statistical or neural models; add physics where the data support it.

## Level 1 household grouping and PV identification

### Group households only if grouping improves the forecasts

Begin with a global model that uses household metadata and known PV status. Separate group models fragment the available sample, so compare them with the global model before adding complexity.

For interpretable clustering, represent each household using past weekday and weekend load shapes, seasonal consumption, heat pump share, variability, peak timing, weather sensitivity, and suitable building features. Scale shape and magnitude separately. Feature clustering using k-means or a mixture is a useful first candidate; the number of groups should be chosen using future validation accuracy and stability.

k-Shape emphasizes normalized shape similarity; DTW and soft-DTW allow temporal displacement. That is useful for discovering household routines, but peak displacement is economically meaningful for procurement. Avoid allowing unrestricted time warping to erase it. Shape normalization can also erase consumption scale unless scale features are retained. [k-Shape](https://www.cs.columbia.edu/~gravano/Papers/2015/sigmod2015.pdf), [soft-DTW](https://proceedings.mlr.press/v70/cuturi17a.html).

Fit clusters on the historical training interval, assign households using information available at each origin, and report a global-versus-grouped ablation. A high silhouette score does not prove improved forecasts. With only 410 households, shared-model group effects are often a more economical experiment than many isolated neural networks; that is a proposed design choice.

### PV ownership inference

Known Installation_HasPVSystem values are labels. Missing survey values mean unknown, not false. If labels are sufficient, train a calibrated classifier such as CatBoost or regularized logistic regression using aggregated historical features. Test on entirely held-out households; splitting daily rows from the same household across train and test produces an overly easy ownership task.

Hu et al. (2021) provide a directly relevant method combining interpretable load features, dimensionality reduction, and clustering to characterize PV and non-PV households. It supports the usefulness of daytime patterns but does not establish that PV is reliably identifiable from this dataset's particular meter channels. [Paper and author manuscript](https://strathprints.strath.ac.uk/78007/).

A daytime dip may reflect PV self-consumption, occupancy, tariffs, heat pump scheduling, or another cause. If the channel records gross use rather than grid import, PV may have little direct signature. Use precision-recall, sensitivity, specificity, probability calibration, and a low-confidence abstention option. Evaluate whether inferred ownership actually improves the consumption forecast. Confirm label timing if installations changed during the data period.

## PV production and net demand if more data become available

This part cannot be validated with the supplied signals. PV ownership alone is not a production time series. Gross load and PV generation cannot generally be uniquely recovered from one observed import series without additional assumptions or measurements.

If actual PV generation, grid export, capacity, orientation, and installation information become available, a strong practical candidate is a physical weather-to-power model plus learned residual correction and quantile or ensemble outputs. pvlib's ModelChain implements the physical path from weather and system configuration to PV power. [Official pvlib documentation](https://pvlib-python.readthedocs.io/en/v0.13.1/reference/modelchain.html).

Day-ahead production needs irradiance and cloud forecasts, not merely temperature. Enforce zero generation at night and plausible capacity bounds, distinguish power from interval energy, and evaluate residual correction against the physical forecast. This is a proposed extension; no reviewed evidence proves it is absolutely best for unknown rooftop configurations.

Do not subtract a synthetic PV estimate from measured grid import if the meter already reflects self-consumption. For net demand, model the joint uncertainty of load and generation rather than treating weather-driven errors as independent.

## Levels 2 and 3 uncertainty estimation and aggregation

### Predict quantiles before choosing purchases

Quantile boosting is the most straightforward first method. Quantile GAMs and quantile regression forests are strong alternative distribution estimators, with different smoothness and pooling behavior. A quantile forest estimates conditional distributions nonparametrically. [Quantile regression forests](https://www.jmlr.org/papers/v7/meinshausen06a.html).

Use several quantiles spanning central and relevant upper tails, then refine the grid around cost-sensitive operating points. Enforce noncrossing quantiles and appropriate physical bounds. A quantile ensemble or QRA can combine different point forecasts into predictive quantiles; train that combination on genuinely out-of-sample historical predictions. [QRA study](https://prac.im.pwr.edu.pl/~hugo/RePEc/wuu/wpaper/HSC_15_01.pdf).

Neural likelihood models, Bayesian neural networks, and dropout-based ensembles are possible competitors. Their uncertainty is not automatically calibrated and may not represent the full randomness of household activity. Validate their distributions with proper scores rather than assuming a particular method gives trustworthy confidence bands.

### Calibrate for time dependence and drift

Conformalized quantile regression uses calibration residuals to adjust learned interval endpoints. Classical finite-sample marginal coverage relies on exchangeability, which household time series with weather dependence and optimization visits generally do not satisfy. [CQR](https://arxiv.org/abs/1905.03222).

Adaptive conformal inference targets long-run coverage under changing distributions. Sequential predictive conformal inference models the changing quantiles of residuals and establishes results under its stated assumptions. Neither provides an unconditional promise that tomorrow's interval is correct for every household, cold spell, and lead time. [Adaptive conformal inference](https://arxiv.org/abs/2106.00170), [SPCI](https://proceedings.mlr.press/v202/xu23r.html).

For this task, begin with recent held-out calibration days, calculate coverage by horizon and important subgroups, and refresh only after the relevant outcomes have become observable. A completed target day should not influence its own issued forecast. Pool horizons or households when data are too scarce for stable calibration, and assess weather-day dependence when computing uncertainty in coverage statistics.

A 2023 Applied Energy study specifically combines CQR with residential total and thermal demand forecasting. It strengthens the application case, although it concerns hourly community forecasting and component inference. Here components are already measured, so there is no need to recreate an unobserved heat-pump channel through causal disaggregation. [Study](https://www.sciencedirect.com/science/article/pii/S0306261923011479).

Interval calibration does not calibrate an entire predictive distribution automatically. A corrected 90% interval does not by itself prove that the 80th percentile is the correct procurement quantile. Validate the quantiles actually used for purchasing. Also distinguish pointwise interval coverage from simultaneous coverage of the entire day's path.

### Forecast the portfolio directly and through households

Compare three methods: sum household conditional means; forecast aggregate load directly; and combine or reconcile forecasts at several levels. Conditional means add exactly. Medians and other marginal quantiles generally do not.

Households share weather and calendar effects. Independent sampling can seriously underestimate portfolio uncertainty. Adding all household 90th percentiles corresponds to perfect rank alignment under a comonotonic construction; it is neither a calibrated portfolio 90th percentile nor a general worst-case guarantee. As an illustration, with 410 equal-variance errors and common pairwise correlation 0.05, total standard deviation is sqrt(1 + 409 × 0.05), or about 4.63 times the independent-error value. This is arithmetic illustrating dependence, not an estimate for this dataset.

For a simple portfolio distribution, calibrate aggregate forecast residuals directly. For paths and household outputs, resample synchronized whole-day residual blocks from comparable earlier days, preserving observed co-movement and temporal structure. Use joint dates rather than sampling a separate date per household. Because coverage varies, explicitly manage the active cohort and missing residuals; never confuse fewer reporting households with lower demand.

Empirical copulas or a Schaake-style reordering can connect calibrated marginal forecasts using historical rank dependence. The Schaake evidence originates in weather postprocessing, so transfer to household residual scenarios needs validation. [Similarity-based Schaake shuffle](https://arxiv.org/abs/1507.02079). Conditional normalizing flows are a more ambitious model for joint energy scenarios, supported by Dumas et al.'s energy forecasting experiments. They require enough data and careful validation rather than automatic trust in visually realistic trajectories. [Scenario modeling with flows](https://arxiv.org/abs/2106.09370).

### Coherence and reconciliation

MinT reconciliation combines independently forecast levels using an estimated error covariance so that forecasts obey aggregation identities. Its optimality is relative to stated linear and covariance assumptions, not all nonlinear procurement losses. A shrinkage covariance and nonnegative constrained version may be needed in practice. [Author treatment of MinT](https://otexts.com/fpp3/reconciliation.html).

If household means are simply added, they are already coherent. Reconciliation is useful when adding separate forecasts for portfolio totals, weather-station groups, or components. For distributions, reconcile joint samples or use a genuine probabilistic reconciliation method; independently modifying quantiles does not generally create a coherent joint distribution. [Probabilistic reconciliation](https://www.sciencedirect.com/science/article/pii/S0377221722006087).

Avoid a full cross-classification of station, PV status, survey class, components, and all temporal totals on the first day. It increases the covariance estimation burden. Begin with the portfolio and the household outputs actually required.

## Evaluation that can identify the best model here

### Prevent leakage in the full pipeline

Use common calendar cutoffs across households. A separate 80/20 split by household can let a global model train on another household's observations after the first household's test origin. For each origin, training targets from every series must already be known. Historical windows may legitimately overlap training and validation; target labels must not cross the origin.

Use older training data, later tuning data, later calibration data, and a final untouched test period. Within the development period, evaluate multiple rolling origins and seasons. Deployable retraining or calibration during the final period is valid only when the policy is fixed beforehand and uses outcomes observed before each new origin. [Rolling-origin evaluation](https://otexts.com/fpp3/tscv.html).

Fit household encodings, clusters, imputation models, transformations, ensemble weights, and thresholds inside each training window. Never fit a full-series decomposition or smoother before splitting. For pretrained models, additionally audit pretraining overlap and distinguish use of external data from a fully past-only trained system.

### Score the outputs that matter

| Output | Main measures | Diagnostic measures |
|---|---|---|
| Household point forecast | MAE in kWh; per-household MASE where its scale is meaningful | RMSE, bias, subgroup performance |
| Portfolio interval energy | MAE, RMSE, underpurchase and overpurchase kWh | High-demand periods and peak timing |
| Daily total | Absolute daily energy error and signed bias | Error by season and weather regime |
| Quantile forecasts | Pinball loss at required quantiles | Calibration curves and quantile crossing |
| Predictive distributions | CRPS or a clearly defined quantile approximation; WIS for interval sets | Coverage together with sharpness |
| Joint day scenarios | Energy score and variogram score | Path, aggregate-tail, and temporal-dependence checks |
| Purchasing policy | Total realized cost; regret relative to stated perfect-information reference | Tail cost, sensitivity to price assumptions |

MAPE is unstable when household or heat-pump use is near zero. MASE provides a scale-aware alternative but can also be undefined for constant series; document its denominator and exceptions. WAPE is useful for aggregate reporting when the denominator is positive, but large users dominate it and signed net-demand cancellation can make it inappropriate. [Hyndman and Koehler on accuracy measures](https://p5g.robjhyndman.com/publications/another-look-at-measures-of-forecast-accuracy/index.html).

Proper scoring rules reward honest distributions. Report interval width together with coverage: very wide intervals can achieve impressive coverage without being useful. The energy score alone may miss incorrect dependence, so a variogram score is informative for joint forecasts. [Proper scoring rules](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf), [variogram scoring](https://repository.library.noaa.gov/view/noaa/22327/).

Compare models using paired errors on the same days and cohort. Report uncertainty with whole-day or longer block resampling appropriate to the observed dependence, rather than treating all quarter-hours and households as independent. Small differences should not be presented as decisive. Check PV status, short histories, weather stations, optimization regimes, and cold days; also report forecast availability and computation time.

## Procurement models and the optimal quantile

Let Y(t) be portfolio demand and q(t) the day-ahead purchase in one delivery interval. For a simple stated model, let c_under > 0 be the incremental cost of underbuying and c_over > 0 the incremental cost of overbuying, relative to purchasing the realized amount day ahead. Then

L(q,Y) = c_under × max(Y − q, 0) + c_over × max(q − Y, 0).

The risk-neutral minimizer is the conditional quantile

q* = F_Y⁻¹(tau), where tau = c_under / (c_under + c_over).

For example, if the assumed underbuying penalty is four times the overbuying penalty, buy the conditional 80th percentile. This ratio is illustrative, not an assertion about real E.ON settlement prices. The median is optimal under equal absolute penalties; the conditional mean is optimal for squared error. Neither is automatically optimal for purchasing.

If p_DA is the day-ahead price, p_buy the price for covering a shortfall, and p_sell the revenue from liquidating surplus, the total cost is

C(q,Y) = p_DA × q + p_buy × max(Y − q, 0) − p_sell × max(q − Y, 0).

When these prices are deterministic at decision time and p_buy > p_DA > p_sell, subtracting the perfect-information day-ahead cost p_DA × Y gives the previous loss with c_under = p_buy − p_DA and c_over = p_DA − p_sell. Prices expressed per MWh need conversion because the demand targets are kWh.

Retail procurement research explicitly connects this type of integrated forecast-and-purchase problem with quantile regression. The wider decision-focused literature motivates training or selecting forecasts using downstream decisions. [Joint energy and smart-meter market](https://arxiv.org/abs/2412.07688), [Smart Predict then Optimize](https://arxiv.org/abs/1710.08005).

If prices are random and correlated with demand, the fixed penalty-ratio rule generally does not solve the full problem. Sample joint demand and price scenarios and optimize expected cost. If storage, heat-pump control, ramping, risk limits, or block orders link intervals, use constrained scenario optimization, optionally with a tail-risk objective such as CVaR. For separable linear costs with no cross-interval constraints, marginal quantiles are sufficient; a giant joint optimizer would add little value.

The challenge supplies no market price series. Therefore use clearly labeled assumptions, report sensitivity across several under/over cost ratios, and call the results simulated procurement savings. Optimize the portfolio buying quantity instead of independently buying every household's upper quantile.

## Recommended experiments and hackathon priorities

### A complete one day solution

1. Establish timestamp, target, completeness, active-cohort, and forecast-cutoff rules. Prepare one common set of historical forecast origins.
2. Build seasonal baselines and a modest ensemble of them.
3. Fit a global direct LightGBM or CatBoost model with mean and quantile predictions. Use past consumption, calendar, known metadata, and only legitimate weather inputs.
4. Add Chronos-2 as a pretrained challenger if setup and inference fit the time budget. If GPU access and a reliable implementation already exist, add a global encoder decoder Transformer instead of spending the entire day tuning several architectures.
5. Compare direct Total with HeatPump plus Other, and direct portfolio with summed household means. Keep the version that wins validation.
6. Calibrate portfolio uncertainty using completed prior days, then simulate cost-based purchases under several stated penalty ratios.
7. Report an untouched temporal test, subgroup results, and a clear next-day demand visualization with uncertainty and buying quantity.

### A deeper research benchmark

Compare the baseline ensemble, pooled analogues, GAM, global LightGBM, global CatBoost, global Transformer, TFT, N-HITS, and selected foundation models. Allocate a comparable tuning budget and report hardware and total pipeline time. Fine-tuned foundation models are a separate condition from frozen zero-shot models.

Run ablations for weather availability; metadata; PV flag; grouping; decomposition; normalization; context length; retraining cadence; calibration method; direct versus household aggregate forecasting; and ensemble combination. An ablation should answer a specific uncertainty, not multiply models without purpose.

Select the forecasting model on validation using distributional or point metrics appropriate to its output. Select the purchasing policy using the agreed cost model. Preserve the final test for one evaluation of the chosen design and predeclared updates.

The most valuable research result might be that a simple pooled model plus calibrated purchasing beats a larger model under the same information set. It might also be that a trained Transformer materially improves cold-weather portfolio forecasts. Both are useful outcomes if demonstrated honestly.

## Questions to settle with mentors

- Does Total measure grid import, gross use, or another boundary, and do the three channels reconcile?
- At what exact time must the next-day forecast be issued, and how delayed are meter observations?
- Which local timezone and delivery intervals define the scored day?
- Are archived weather forecasts available, or only realized station observations?
- Are household and survey labels valid throughout history, particularly PV installation and optimization dates?
- Will the evaluator score household accuracy, aggregate demand, simulated procurement cost, or a combination?

## Selected references and implementation entry points

The following reading list is grouped by its use in the project. Links point to original research or official resources. Dates distinguish papers from later releases where relevant.

### Closely matched load forecasting studies

1. Semmelmann et al. 2024. **The impact of heat pumps on day-ahead energy community load forecasting.** Applied Energy. [Author-hosted paper](https://publikationen.bibliothek.kit.edu/1000170854/152908535).
2. Botman et al. 2025. **A global probabilistic approach for short-term forecasting of individual households electricity consumption.** Applied Energy. [Author manuscript](https://ftp.esat.kuleuven.be/pub/pub/stadius/ida/reports/24-178.pdf).
3. Hertel et al. 2026. **A Benchmark for Electrical Load Forecasting Across Grid Levels.** Preprint. [Paper](https://arxiv.org/abs/2607.15705), [code](https://github.com/KIT-IAI/load-forecasting-benchmark).
4. Emami, Sahu, and Graf. 2023. **BuildingsBench.** NeurIPS Datasets and Benchmarks. [Paper](https://arxiv.org/abs/2307.00142), [benchmark](https://github.com/NatLabRockies/BuildingsBench).
5. Ben Taieb et al. 2016. **Forecasting Uncertainty in Electricity Smart Meter Data by Boosting Additive Quantile Regression.** IEEE Transactions on Smart Grid. [Author page](https://souhaib-bentaieb.com/publications/smart-meter-quantiles/).
6. Gaillard, Goude, and Nedellec. 2016. **Additive models and robust aggregation for GEFCom2014 probabilistic electric load and electricity price forecasting.** International Journal of Forecasting. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0169207015001545).
7. **Total and thermal load forecasting in residential communities through probabilistic methods and causal machine learning.** 2023. Applied Energy. [Paper](https://www.sciencedirect.com/science/article/pii/S0306261923011479).
8. Chattopadhyay et al. 2026. **Peak-Aware Short-Term Load Forecasting Across Distribution Grid Aggregation Levels.** arXiv version; listed as accepted at IEEE PES ISGT Europe 2026. [Paper](https://arxiv.org/abs/2609.18588).
9. Pendyala et al. 2026. **Assessing Covariate-Informed Grid Load Forecasting with a Time-Series Foundation Model.** Preprint. [Paper](https://arxiv.org/abs/2609.06656).

### Forecasting model foundations

10. Montero-Manso and Hyndman. 2021. **Principles and algorithms for forecasting groups of time series: Locality and globality.** International Journal of Forecasting. [Preprint](https://arxiv.org/abs/2008.00444).
11. Ke et al. 2017. **LightGBM: A Highly Efficient Gradient Boosting Decision Tree.** NeurIPS. [Paper](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html).
12. Prokhorenkova et al. 2018. **CatBoost: unbiased boosting with categorical features.** NeurIPS. [Paper](https://proceedings.neurips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html).
13. Lim et al. 2021. **Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.** International Journal of Forecasting; preprint 2019. [Paper](https://arxiv.org/abs/1912.09363).
14. Challu et al. 2023. **N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.** AAAI; preprint 2022. [Paper](https://arxiv.org/abs/2201.12886).
15. Nie et al. 2023. **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.** ICLR. [Paper](https://arxiv.org/abs/2211.14730).
16. Wang et al. 2024. **TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables.** [Paper](https://arxiv.org/abs/2402.19072).
17. Salinas et al. **DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks.** Preprint 2017; International Journal of Forecasting version 2020. [Paper](https://arxiv.org/abs/1704.04110).

### Current foundation models and benchmarks

18. Ansari et al. 2025. **Chronos-2: From Univariate to Universal Forecasting.** [Technical report](https://arxiv.org/abs/2510.15821), [official model card](https://huggingface.co/amazon/chronos-2).
19. Google Research. 2026. **TimesFM-3: A zero-shot foundation model for multivariate forecasting.** [Official announcement](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/), [repository and terms](https://github.com/google-research/timesfm).
20. Podest et al. 2026. **TiRex-2: Generalizing TiRex to Multivariate Data and Streaming.** [Preprint](https://arxiv.org/abs/2607.01204), [official model card](https://huggingface.co/NX-AI/TiRex-2).
21. Khwaja et al. 2026. **Toto 2.0: Time Series Forecasting Enters the Scaling Era.** [Technical report](https://arxiv.org/abs/2605.20119).
22. Shchur et al. 2025. **fev-bench: A Realistic Benchmark for Time Series Forecasting.** [Paper](https://arxiv.org/abs/2509.26468), [evaluation software](https://github.com/autogluon/fev).
23. Aksu et al. 2024. **GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation.** [Paper](https://arxiv.org/abs/2410.10393).
24. Qiao et al. 2026. **It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks.** [Paper](https://arxiv.org/abs/2602.12147).

### Structure and preprocessing

25. Du, Côté, and Liu. 2023. **SAITS: Self-attention-based imputation for time series.** Expert Systems with Applications. [Paper](https://www.sciencedirect.com/science/article/pii/S0957417423001203).
26. Omar and Holmberg. 2021. **Load Forecasting Tool for NIST Transactive Energy Market.** [NIST publication](https://www.nist.gov/publications/load-forecasting-tool-nist-transactive-energy-market).
27. Hu et al. 2021. **Classification and characterization of intra-day load curves of PV and non-PV households using interpretable feature extraction and feature-based clustering.** Sustainable Cities and Society. [Author manuscript](https://strathprints.strath.ac.uk/78007/).
28. Paparrizos and Gravano. 2015. **k-Shape: Efficient and Accurate Clustering of Time Series.** SIGMOD. [Paper](https://www.cs.columbia.edu/~gravano/Papers/2015/sigmod2015.pdf).
29. Cuturi and Blondel. 2017. **Soft-DTW: a Differentiable Loss Function for Time-Series.** ICML. [Paper](https://proceedings.mlr.press/v70/cuturi17a.html).
30. ECMWF. 2025. **ECMWF's ensemble AI forecasts become operational.** [Official announcement](https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwfs-ensemble-ai-forecasts-become-operational).
31. pvlib contributors. **ModelChain.** [Official documentation](https://pvlib-python.readthedocs.io/en/v0.13.1/reference/modelchain.html).

### Uncertainty and evaluation

32. Meinshausen. 2006. **Quantile Regression Forests.** JMLR. [Paper](https://www.jmlr.org/papers/v7/meinshausen06a.html).
33. Romano, Patterson, and Candès. 2019. **Conformalized Quantile Regression.** NeurIPS. [Paper](https://arxiv.org/abs/1905.03222).
34. Gibbs and Candès. 2021. **Adaptive Conformal Inference Under Distribution Shift.** NeurIPS. [Paper](https://arxiv.org/abs/2106.00170).
35. Xu and Xie. 2023. **Sequential Predictive Conformal Inference for Time Series.** ICML. [Paper](https://proceedings.mlr.press/v202/xu23r.html).
36. Liu, Nowotarski, Hong, and Weron. **Probabilistic Load Forecasting via Quantile Regression Averaging on Sister Forecasts.** Online 2015; IEEE Transactions on Smart Grid issue 2017. [Author manuscript](https://prac.im.pwr.edu.pl/~hugo/RePEc/wuu/wpaper/HSC_15_01.pdf).
37. Hyndman and Athanasopoulos. **Forecasting Principles and Practice**, third edition, forecast reconciliation and time series cross-validation chapters. [Reconciliation](https://otexts.com/fpp3/reconciliation.html), [evaluation](https://otexts.com/fpp3/tscv.html).
38. Panagiotelis et al. 2023. **Probabilistic forecast reconciliation: Properties, evaluation and score optimisation.** European Journal of Operational Research. [Paper](https://www.sciencedirect.com/science/article/pii/S0377221722006087).
39. Dumas et al. 2021 preprint. **A deep generative model for probabilistic energy forecasting in power systems: normalizing flows.** [Paper](https://arxiv.org/abs/2106.09370).
40. Schefzik. 2015 preprint. **A similarity-based implementation of the Schaake shuffle.** [Paper](https://arxiv.org/abs/1507.02079).
41. Gneiting and Raftery. 2007. **Strictly Proper Scoring Rules, Prediction, and Estimation.** JASA. [Author paper](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
42. Scheuerer and Hamill. 2015. **Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of Multivariate Quantities.** Monthly Weather Review. [NOAA manuscript](https://repository.library.noaa.gov/view/noaa/22327/).
43. Hyndman and Koehler. 2006. **Another look at measures of forecast accuracy.** International Journal of Forecasting. [Author page](https://p5g.robjhyndman.com/publications/another-look-at-measures-of-forecast-accuracy/index.html).

### Purchasing decisions

44. Elmachtoub and Grigas. 2022. **Smart Predict then Optimize.** Management Science; preprint 2017. [Paper](https://arxiv.org/abs/1710.08005).
45. Chhachhi and Teng. 2024 preprint. **A Joint Energy and Differentially-Private Smart Meter Data Market.** [Paper](https://arxiv.org/abs/2412.07688).
