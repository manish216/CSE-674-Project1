Project1:
Determining the Probabilities of Hand Writing Formations Using PGM’S
==========================
Manish Reddy Challamala,
March 8, 2019 ,manishre@buffalo.edu

## Abstract

To develop a Probability Graphical Model (PGM’s) to determine the best
model for the given data and to infer the Maximum a-posterior probability
estimate to obtain unobserved quantity for empirical data.

## 1 Introduction

The goal of the project is to develop a Probabilistic Graphical Model (PGM’s) for the given
handwriting patterns and to find the best Bayesian setting for the given features.

The features are obtained from two different sources:

1. Marginal and Conditional probability tables are provided for the bigram ‘th’.
2. MNIST Dataset.

The code is implemented in five phases:

For bi-gram “th”:

- We need to find the pair-wise correlations and in-dependencies that exists between
    six features. We can find independencies between xi and xj by using the following
    formula
    
    <img src="http://www.sciweavers.org/tex2img.php?eq=abs%28%28P%28x%2Cy%29-p%28x%29%28y%29%29%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="abs((P(x,y)-p(x)(y)))" width="197" height="19" />
    
Here x is xi and y is xj
P(x, y) is Joint Probability.
P(x)P(y) is the product of marginal probabilities.
- We need to construct several Bayesian networks and find the high probability and
    low probability Bayesian network among the networks by using the goodness score
    like K2 score
- Convert the high probability Bayesian network into Markov network using
    moralization and find the time taken by the Bayesian and Markov network to infer a
    query.
For MNIST Dataset:
- For MNIST the sample data for the features f1 to f9 are given and we need to
construct several Bayesian networks to evaluate the goodness score of the network
to determine high probability model.
- Infer the conditional probability distribution tables of the model.


## 2 Theory

**2.1 Bayesian network**

- Bayesian network is probabilistic graphical models for representing the multivariate
    probability distributions, in which nodes represent the random variables and the edges
    represent conditional probability distributions [CPD’s] tables between random
    variables, which are used to calculate the probability dependencies between the
    variables.
- Bayesian networks are also called as belief or causal networks because they are
    directed acyclic graphs [DAG], even with a change of CPD at one node can affect the
    whole networks performance.
- The probability distribution is defined in the form of factors:

<img src="http://www.sciweavers.org/tex2img.php?eq=P%28%20X_%7B1%7D%2BX_%7B2%7D%2BX_%7B3%7D%2B...%2BX_%7BN%7D%20%29%20%3D%20%20%5Cprod_%7Bi%3D1%7D%5E%7BN%7D%20P%28X_%7Bi%7D%20%7C%20%5Cprod%20X_%7Bi%7D%29%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="P( X_{1}+X_{2}+X_{3}+...+X_{N} ) =  \prod_{i=1}^{N} P(X_{i} | \prod X_{i}) " width="347" height="53" />

Where <img src="http://bit.ly/2UQ1PSl" align="center" border="0" alt="\prod_{i=1}^{N} P(X_{i} | \prod X_{i}) " width="119" height="53" />
is P (node | parent(node))

**2 .1 Markov network**

- Markov network is undirected acyclic graphs which are similar to the Bayesian
    network. As the graphs are undirected, instead of edges they have cliques which
    connect each node with their neighboring nodes.
- The probability distribution is defined in the form of factors of potential functions.
    Which can be written in the log-linear model.
- As there is no topological order for the Markov network, we don’t use the chain rule
    instead of potential functions for each clique in the graph.
- Joint distribution in the Markov network is proportional to the product of clique
    potentials
- The conditional probability is defined as

```
푃(푦 |휃)=
```
### 1

### 푍(휃)

### ∏휙(푦푐 | 휃푐)

```
푐
```
```
Where 푍(휃) is the partition function derives as summation of products of all
potential factors.
```
**2.2 Moralization**

- Moralization is the method to get the moral graphs, which finds the equivalent
    undirected graph form a directed graph.
- Moralization is adding the edges between the unlinked nodes of a parent which have
    the common child.


## 3 Experimental Setup:

**3.1 Task1: Finding Pair-wise Correlations and Independencies**

We are calculating the pair-wise correlations and independencies for all possible combinations
of xi and xj.

```
Correlation = Σ((푎푏푠 푃 (푥푖,푥푗) −푃 (푥푖) 푃( 푥푗))
```
Where 푃 (푥푖,푥푗) is the joint probability distribution.

푃 (푥푖)푃( 푥푗) is the product of marginal distributions.

The correlation values are stated below:

Table 1: Pair-wise Correlation values and Independencies.

GIVEN


 |  val   |   X1   |   X2   |   X3   |   X4   |   X5   |  X6    | 
 |  :---: |  :---: |  :---: |  :---: |  :---: | :---:  |  :---: |
 |   X1   |    0   |    0   |    0   | 0.119  |   0    | 0.1666 |
 |   x2   |  0.159 |    0   | 0.2982 | 0.115  |  0.36  | 0.1749 |
 |   x3   |    0   |  0.218 |    0   |    0   | 0.115  | 0.0948 | 
 |   x4   | 0.119  |    0   |   0    |    0   |   0    | 0.1432 | 
 |   x5   |    0   | 0.129  | 0.1156 |    0   |    0   |    0   | 
 |   x6   |  0.160 |    0   | 0.094  | 0.094  |    0   |    0   | 

- By placing the threshold value as 0.15 we can find strong and weak correlation
    relationship between the nodes. Lower the value the higher the correlation.
- Strong Correlations: (x 4 | x 1 ), (x 5 | x 2 ), (x 6 | x 3 ), (x 6 | x 4 ), (x 3 | x 5 ), (x 3 | x 6 )
- Weak Correlations: (x 6 | x 1 ), (x 3 | x 2 ), (x 2 | x 3 ), (x 1 | x 4 ), (x 2 | x 5 ), (x 2 | x 6 )
- From the above values we can infer that (x 6 , x 3 ) are strongly correlated in bi
    directional way and these values are dependent on each other.
- Similarly, for the weak correlations the x 2 and x 3 values are independent to each other
    and we can also observe that node x2 is appearing as a weak correlation node for most
    of the other nodes and can conclude that x 2 node as good correlation with x 1 node
    when compared to other nodes.
**3. 2 Task2: Constructing the Bayesian networks and finding the goodness score**
- In this task, we are constructing the five Bayesian networks using the threshold value
calculated from the above task1.
- Creating a Bayesian network by using ‘pgmpy. models’ – Bayesian Model function
and adding the nodes and corresponding edges to the model.
- Adding the CPD tables of the edges using ‘Tabularcpd’ -add_cpds function.
- After instantiating the model object, we are performing the Ancestral or forward
sampling.


**3.2.1 Ancestral or Forward Sampling**

- Given a Bayesian network with probability distributions of nodes x1 to x6, we sample
    each variable in topological order.
- Start sampling the nodes with no parents and take the index of the highest value of
    that node and sample the next child nodes based on the sampled values at the first
    step.
- To sample a discrete distribution we split [0,1] interval into bins whole sizes are
    determined by the probabilities P(xi), I = 1,2,3...,k
- Generate a sample s uniformly from the interval
- If S is the ith interval then sampled value is xi
- After generating the sample data for the five models we are combining the five data
    sets and calculating the goodness score for by using the k2 function.

```
Table2: K2 Scores for Bayesian Models
```
- By comparing the goodness score of the five Bayesian models we can observe that
    model 4 is the High probability model where the parent node is x1 dependent on x
    and x6, the child node x4 is dependent on its descendant x2, which is dependent on
    x5 and the node x5 is connected with x3.
- Edges of high probability model: ('x4','x1'),('x1','x2'),('x1','x6'),('x2','x3'),('x3','x5')
- Edges of low probability model : ('x1','x4'),('x1','x6'),('x4','x2'),('x2','x5'),('x5','x3')

Figure1: Low Probability ‘th’ model Figure2: High Probability ‘th’ model

```
K2 Score Values
```
```
Model 1 - 31949.
```
```
Model 2 - 32018.
```
```
Model 3 - 32090.66 [LOW PROBABILITY MODEL]
```
```
Model 4 - 31947.80 [HIGH PROBABILITY MODEL]
```
```
Model 5 - 32038.
```

## 3 Task 3: Converting Bayesian network to Markov network

## 3.1 Converting to Markov network

- From the above task consider the high probability ‘th’ model and convert it into a
    Markov network using the moralization.
- Model_name.to_markov_model() predefined function is used to convert the Bayesian
    network into a Markov network.

**3.2 Inferring the query**

- Some conditional probability inference algorithms are stated below:
    1. Variable Elimination
    2. Belief Propagation
    3. Variational Approximations
    4. Markov chain Monte Carlo (MCMC)
- What is meant by inference in Bayesian network?
    Given a query (joint probability distribution) we are getting the probabilities for all
    other specified nodes whose sum should be 1.
- In this process belief propagation and Variable elimination, algorithms have been used
    to infer a single query from both Bayesian and Markov networks and time taken to
    infer the query is calculated.
- The same query id inferred multiple times and time is compared for both networks
    and count will generate each time
- From the multiple results, I can infer that time taken for Bayesian network is greater
    than Markov network for both variable and belief propagation methods
- The average time taken for Bayesian network to infer the query is 0.046 sec and for
    Markov network is 0.062 sec.
- The query inferred for belief propagation is
    variables=['x1','x3','x4','x5'], evidence= {'x2': 2, 'x6': 0}
- The model returned the following probabilities
    The query is: {'x1': 0, 'x3': 1, 'x4': 0, 'x5': 0}
- For the given evidence parameters
    풙ퟐퟐ : shape of loop of h is curved left side and straight right side
    풙ퟔퟎ : shape of t is tented.
- The model returned the best settings as :
    풙ퟏퟎ : Height relationship of t to h where t is shorter than h
    풙 31 : Shape of Arch of h is pointed
    풙 40 : Height of cross is on upper half of staff
    풙ퟓퟎ: Base line of h is slanting upwards

Figure3: Markov model of best probability ‘th’ mode


## 4 Task 4: Constructing Bayesian networks for MNIST

## dataset

**4.1 Constructing the Bayesian network**

- In this task, we need to construct the Bayesian network, by using the data given
    One model is built by using the hill climb search method and the second model is
    built by using the constraint-based estimator which gives us a partial network and
    remaining nodes are constructed arbitrarily.

**4.2 Evaluating the goodness Score of the models**

- The goodness score of all models are calculated using the k2 score and the results
    are compared to find the high probability ‘AND’ model and low probability ‘AND’
    model which are displayed below.


Table3: K2 Score for ‘AND’ models

| K2 SCORE   | VALUES |
| :---:      | :---:  |
| Model 1    | -9462.[HIGH PROBABILITY AND MODEL] |
| Model 2    | -9651.[LOW PROBABILITY AND MODEL]  |
| Model 3    | -9650. |
| Model 4    |-9638.  |
| Model 5    |-9708.  |




