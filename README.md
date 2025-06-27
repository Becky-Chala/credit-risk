## Credit Scoring Businedd Understanding

### 1. ●	How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
Basel II Accord is like a big international rulebook for banks. It says that banks must carefully measure and explain the risk when they are giving loans.
Because of this, credit risk modle needs to be:

●  Easy for people (and government regulators) to understand

●  Clear about how it makes decisions

●  Well documented (so we can show step-by-step how it works)

That’s why we should choose a model that is not just good at making predictions but also easy to explain and justify.

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
There is no column in the data that have the persons list that failed to pay back their loans and we still need to traind a model so we create a substitute (proxy) variable.
We will look at customer behaviors like how much they spend, form this there is a chance of labeling the customers as:

●  High risk (default service which means no loans)

●  Low risk (likely to repay so there is a chance of getting a loan)

Proxy might not be perfect and we might get the label wrong switching the good customers with the bad ones, which leads to wrong decisions.

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

Simple models (like Logistic Regression with WoE):
Pros:
Easy to understand
Easy to explain to managers and regulators
You can see which factors (features) affect the result

Cons:
Not always the most accurate
Might miss complicated patterns

Complex models (like Gradient Boosting Machines - GBM):
Pros:
More accurate
Better at finding hidden patterns in data

Cons:
Hard to explain
Regulators may not trust it
❌ Can be risky if we don’t fully understand how it makes decisions
