---
geometry: margin=3cm
fontsize: 12pt
---

# Chromosome segregation model - 1D version

_(11/04/2014)_

## Introduction

This is a more detailed version of the kinetochore segregation model to be published in the JCB article, which should be referred to for all the experimental, biological and non-technical aspects of this work.

## Definitions

### State vector

The mitotic spindle is described by the speeds and positions along the $x$ axis of two spindle pole bodies, $N$ chromosomes with two centromeres and $M_k$ attachment sites per centromere.

Positions are noted as follows:

- The left and right spindle pole bodies ( SPBs ), $x_s^L$ and $x_s^R$
- The $N$ centromeres, $x_n^A, \, x_n^B, n \in \{1,\cdots, N\}$
- The $M_k$ attachment sites of each centromere, $x_{nm}^A, \,
  x_{nm}^B, n \in \{1, \cdots, N \}, m \in \{1, \cdots, M_k\}$

The speeds are noted with a dot: $dx / dt = \dot{x}$.

As all the interactions are assumed to be parallel to the spindle axis, only the positions along this axis are considered, in a coordinate system with its origin at the center of the spindle, which means that $x_s^L(t) = - x_s^R(t)\, \forall t$.

### Random variables for the attachment

We define $\rho_{nm}^A$ and $\lambda_{nm}^A$, two random variables that define the attachment state of the site $x_{nm}^A$, such that:

\begin{align}
  \label{eq:rholambda}
  \lambda_{nm}^A &=
  \begin{cases}
    1 &\text{if the site is attached to the left SPB}\\
    0 &\text{otherwise}\\
  \end{cases}\\
  \rho_{nm}^A &=
  \begin{cases}
    1 &\text{if the site is attached to the right SPB}\\
    0 &\text{otherwise}\\
  \end{cases}
\end{align}

Note that $\rho_{nm}^A$ and $\lambda_{nm}^A$ are not independent, as an attachment site can't be attached to both poles. To take this into account, we can define the variable $\pi_{nm}^A = \rho_{nm}^A - \lambda_{nm}^A$ such that:

\begin{equation}
  \label{eq:pnm}
  \pi_{nm}^A =
  \begin{cases}
    - 1 &\text{if the site is attached to the left SPB}\\
    0 &\text{if the site is not attached}\\
    1 &\text{if the site is attached to the right SPB}\\
  \end{cases}\\
\end{equation}

We have:

\begin{align}
  \lambda_{nm}^A &= \pi_{nm}^A\left(\pi_{nm}^A - 1\right)/2\\
  \rho_{nm}^A &= \pi_{nm}^A\left(\pi_{nm}^A + 1\right)/2
\end{align}

We also define $N_n^{AL}$ and $N_n^{AR}$ as the number of ktMTs of
centromere A attached to the left and right SPBs, respectively:

\begin{equation}
  \label{eq:NAL}
  N_n^{AL} = \sum_{m = 1}^{M_k}\lambda_{nm}^A \mbox{ and }%
  N_n^{AR} = \sum_{m = 1}^{M_k}\rho_{nm}^A
\end{equation}

Note that $N_n^{AL} + N_n^{AR} \leq M_k \forall\ \pi_{nm}$. The same definitions apply for the centromere B and left SPB.

## Mechanical system

### Forces

The following force balances are considered:

#### Forces at the right SPB

- Friction forces (viscous drag):  $F_f^R = -\mu_s \dot{x_s}^R$

- Midzone force generators:
  $$F_{mid} = F_z\left(1 - (\dot{x}^R_s - \dot{x}_s^L)/V_z\right) =
  F_z\left(1 - 2\dot{x}^R_s / V_z\right) $$

- Total kinetochore microtubules force generators:

 $$ \begin{aligned}
    F_{kMT}^T = \sum_{n = 1}^{N}\sum_{m = 1}^{M_k} & - \rho_{nm}^A\,F_k\left( 1 -
      (\dot{x}^A_{nm} - \dot{x}^R_s)/V_k\right)\\
    & - \rho_{nm}^B\,F_k\left( 1 -
      (\dot{x}^B_{nm} - \dot{x}^R_s)/V_k\right)\\
 \end{aligned} $$

#### Forces at the left SPB

Because of the reference frame definition, $\dot{x_s}^R = -\dot{x_s}^L\,\forall t$. Here we substituted $x_s^L$ with $-x_s^R$

- Friction forces (viscous drag):  $F_f^L = \mu_s \dot{x_s}^R$

- Midzone force generators:
  $$F_{mid}^L = - F_z\left(1 - 2\dot{x}^R_s / V_z\right) $$

- Total kinetochore microtubules force generators:
$$  \begin{aligned}
    F_{kMT}^T = \sum_{n = 1}^{N}\sum_{m = 1}^{M_k} &  \lambda_{nm}^A\,F_k\left(1 +
      (\dot{x}^A_{nm} + \dot{x}^R_s)/V_k\right)\\
    & \lambda_{nm}^B\,F_k\left(1 +
      (\dot{x}^B_{nm} + \dot{x}^R_s)/V_k\right)
  \end{aligned}
$$

#### Forces at centromere $An$

- Drag: $F_c^f = -\mu_c \dot{x_n}^A$

- Cohesin bond (Hook spring) restoring force exerted by centromere\footnote{We want the centromeres to be able to cross each other. In one dimension, this introduces a discontinuity. In the previous version, the 'swap' mechanism was solving this directly (as $x_A$ and $x_B$ are exchanged). This is not possible any more, as the 'swap' mechanism is now irrelevant, as there is no prefered side for a given centromere.}:

  \begin{equation}
    F_{BA} =
    \begin{cases}
      \kappa_c (x_n^B - x_n^A - d_0) &\mathrm{if}\quad x_n^A \leq x_n^B\\
      %0  &\mathrm{if}\quad - d_0 < x_n^A - x_n^B < d_0\\
      \kappa_c (x_n^B - x_n^A + d_0) &\mathrm{if}\quad  x_n^A > x_n^B\\
    \end{cases}
  \end{equation}
  With $F_{AB} = - F_{BA}$.

- Total visco-elastic bond between the centromere A and the attachment sites (Note that the rest length of the spring is put to 0) :

  $$ F_v^T = \sum_{m = 1}^{M_k} -\kappa_k(x_n^A - x_{nm}^A)
  - \mu_k(\dot{x}_n^A - \dot{x}_{nm}^A) $$

#### Forces at attachment site $Anm$

- Visco-elastic bond between the centromere A and the
  attachment sites:

  $$F_v =  \kappa_k(x_n^A - x_{nm}^A)
  + \mu_k(\dot{x}_n^A - \dot{x}_{nm}^A) $$

- Kinetochore microtubules force generators:
  \begin{equation}
    \begin{aligned}
      F_{kMT}^A &= F_{kMT}^{RA} + F_{kMT}^{LA}\\
      F_{kMT}^{RA} &= \rho_{nm}^A\,F_k\left(1 - \frac{\dot{x}^A_{nm} -
          \dot{x}^R_s}{V_k}\right)\\
      F_{kMT}^{LA} &=  \lambda_{nm}^A\,F_k\left(-1 - \frac{\dot{x}^A_{nm} -
          \dot{x}^L_s}{V_k}\right)\\
    \end{aligned}
  \end{equation}

For now on, we are taking $F_k$ as unit force and $V_k$ as unit speed (i.e $F_k = 1$ and $ V_k = 1$), this gives:

\begin{equation}
 F_{kMT}^A = \rho_{nm}^A\,\left(\dot{x}^R_s - \dot{x}^A_{nm} + 1\right)%
 - \lambda_{nm}^A\,\left(\dot{x}^R_s + \dot{x}^A_{nm} + 1\right)
\end{equation}

Eventually, substituting $\rho^A_{nm} - \lambda^A_{nm}$ with $\pi_{nm}^A$ and $\lambda^A_{nm} + \rho^A_{nm}$ with $|\pi_{nm}^A|$:

\begin{equation}
    F_{kMT}^A =  \pi_{nm}^A(\dot{x}^R_s + 1) - |\pi_{nm}^A|\dot{x}^A_{nm}
\end{equation}

### Set of coupled first order differential equations

In the viscous nucleoplasm, inertia is negligible. Newton first principle thus reduces to: $\sum F = 0$ on each object of the spindle. This equation of motion can be written for each element of the spindle of coordinates $x_s^R, x_s^L, x_n^A, x_{nm}^A,  x_n^B$ and $x_{nm}^B$.

- Equation of motion at left/right SPBs

To simplify further, the equations for the right and left SPBs can be
combined:

\begin{equation}
  \begin{aligned}
    - \mu_s\dot{x}^R_s + F_{z}\left(1 - 2\dot{x}^R_s/V_z\right)%
    + \sum_{n,m} - \rho_{nm}^A\,\left(\dot{x}^R_s - \dot{x}^A_{nm} +%
      1\right) - \rho_{nm}^B\,\left(\dot{x}^R_s - \dot{x}^B_{nm} +%
      1\right)&= 0 \, \mbox{for the right SPB}\\
    \mu_s\dot{x}^R_s - F_{z}\left(1 - 2\dot{x}^R_s/V_z\right)%
    + \sum_{n,m} -\lambda_{nm}^A\,\left(\dot{x}^R_s + \dot{x}^A_{nm} +%
      1\right) - \lambda_{nm}^B\,\left(\dot{x}^R_s + \dot{x}^B_{nm} +%
      1\right) &= 0 \, \mbox{for the left SPB}\\
  \end{aligned}
\end{equation}

The difference of those two expressions gives, with the same substitutions as before:

\begin{equation}
  \label{eq:spindle_term}
  - 2\mu_s\dot{x}^R_s + 2F_{z}\left(1 - 2\dot{x}^R_s/V_z\right)%
  + \sum_{n,m}- (|\pi_{nm}^A|  + |\pi_{nm}^B|)(\dot{x}^R_s + 1)%
  + \pi_{nm}^A \dot{x}_{nm}^A + \pi_{nm}^B \dot{x}_{nm}^B= 0%
\end{equation}

- Equation of motion at each centromere $A_n$

\begin{equation}
-\mu_c \dot{x_n}^A+\kappa_c (x_n^B - x_n^A - \delta_n d_0) + \sum_{m = 1}^{M_k} -\kappa_k(x_n^A - x_{nm}^A)
  - \mu_k(\dot{x}_n^A - \dot{x}_{nm}^A) = 0  %
    \mbox{ with }\, \delta_n =%
    \begin{cases}
      1  &\mathrm{if}\quad  x_n^A < x_n^B\\
      -1 &\mathrm{if}\quad  x_n^A > x_n^B\\
    \end{cases}
\end{equation}

- Equation of motion at each attachment site $A_nm$

\begin{equation}
\kappa_k(x_n^A - x_{nm}^A) + \mu_k(\dot{x}_n^A - \dot{x}_{nm}^A) + \pi_{nm}^A(\dot{x}^R_s + 1) - |\pi_{nm}^A|\dot{x}^A_{nm} = 0
\end{equation}

- These 3 equations are gathered together in the system of equations:

$$
\mathbf{A}\dot{X} + \mathbf{B}X + C = 0
$$

The vector $X$ has $1 + 2N(M_k + 1)$ elements and is defined as follow\footnote{Note that the left SPB is omitted in $X$.}:

\begin{equation*}
  X = \{x_s^R, \{x_n^A, \{x_{nm}^A\},  x_n^B,%
  \{x_{nm}^B \}\}\}\mbox{ with } n \in 1 \cdots N %
  \mbox{ and } m \in 1 \cdots M_k
\end{equation*}
In matrix form, we have:\\
\begin{equation}
  \begin{aligned}
    X = &%
    \begin{pmatrix}
      x_s^R\\
      x_n^A\\
      x_{nm}^A\\
      x_n^B\\
      x_{nm}^B\\
    \end{pmatrix} =%
    \begin{pmatrix}
      \text{right SPB}\\
      \text{centromere }A, n\\
      \text{attachment site }A, n,m\\
      \text{centromere }B, n\\
      \text{attachment site }B, n,m\\
    \end{pmatrix}\\
    A = &%
    \begin{pmatrix}
      %%%%% SPB %%%%
      - 2 \mu_s - 4 F_z/V_z - \sum (|\pi_{nm}^A| + |\pi_{nm}^B|)& 0 & \pi_{nm}^A &%
      0 &  \pi_{nm}^B\\


      %%%%% Centromere A %%%%
      0 &  -\mu_c - M_k \mu_k& \mu_k & 0 & 0\\
      %%%%% Att. Site A %%%%
      \pi_{nm}^A & \mu_k & - \mu_k - |\pi_{nm}^A| & 0& 0\\
      %%%%% Centromere B %%%%
      0&0&0 & -\mu_c - M_k \mu_k & \mu_k\\
      %%%%% Att. Site B %%%%
      \pi_{nm}^B & 0&0 & \mu_k & - \mu_k - |\pi_{nm}^B| \\
    \end{pmatrix}, \\
     % = &%
    B = &%
    \begin{pmatrix}
      0 & 0&0&0&0\\
      0 & - \kappa_c - M_k \kappa_k & \kappa_k &%
      \kappa_c & 0 \\
      0 & \kappa_k & -\kappa_k & 0&0\\
      0 & \kappa_c & 0 &%
      -\kappa_c - M_k \kappa_k & \kappa_k \\
      0&0&0 & \kappa_k & - \kappa_k\\
    \end{pmatrix}\\
    C = &%
    \begin{pmatrix}
      2Fz - \sum_{n,m}(|\pi_{nm}^A| + |\pi_{nm}^B|) \\
      - \delta_n \kappa_c d_0\\
      \pi_{nm}^A\\
      \delta_n \kappa_c d_0\\
      \pi_{nm}^B\\
    \end{pmatrix}
    % \begin{cases}
    %   1  &\mathrm{if}\quad d_0 < x_n^A - x_n^B\\
    %   0  &\mathrm{if}\quad - d_0 < x_n^A - x_n^B < d_0\\
    %   -1 &\mathrm{if}\quad  x_n^A - x_n^B\\
    % \end{cases}
\end{aligned}
\end{equation}

As is actually done in the python implementation, $A$  can be decomposed into a time invariant part $A_0$ and a variable part $A_t$ with:

\\
\begin{equation}
  \begin{aligned}
    A_0 &=%
    \begin{pmatrix}
      - 2 \mu_s - 4 F_z/V_z & 0&0&0&0\\
      0 &  -\mu_c - M_k \mu_k& \mu_k & 0&0\\
      0 & \mu_k & - \mu_k & 0&0\\
      0&0&0 & -\mu_c - M_k \mu_k & \mu_k\\
      0&0&0 & \mu_k & - \mu_k\\
    \end{pmatrix}\\
    A_t &=%
    \begin{pmatrix}
      %%%%% SPB %%%%
      - \sum (|\pi_{nm}^A| + |\pi_{nm}^B|)& 0 & \pi_{nm}^A &%
      0 &  \pi_{nm}^B\\
      %%%%% Centromere A %%%%
      0&0&0&0&0\\
      %%%%% Att. Site A %%%%
      \pi_{nm}^A & 0 & - |\pi_{nm}^A| & 0&0\\
      %%%%% Centromere B %%%%
      0&0&0&0&0\\
      %%%%% Att. Site B %%%%
      \pi_{nm}^B & 0&0&0 & - |\pi_{nm}^B| \\
    \end{pmatrix}\\
  \end{aligned}
\end{equation}

For the sake of clarity, $B$ can be decomposed in a kinetochore and a cohesin part, $B = B_c + B_k$:

\begin{equation}
  B = \kappa_k%
  \begin{pmatrix}
    0 & 0&0&0&0\\
    0 &  - M_k  & 1 & 0&0 \\
    0 & 1 & -1 &  0&0\\
    0 &  0&0 & - M_k  & 1 \\
    0&0&0  & 1 & - 1\\
  \end{pmatrix}
  + \kappa_c%
  \begin{pmatrix}
    0 & 0&0&0&0\\
    0 & - 1 & 0 & 1  & 0 \\
    0&0&0&0&0\\
    0 & 1 & 0 & -1 & 0 \\
    0&0&0&0&0\\
  \end{pmatrix}
\end{equation}

## Attachment instability

### Attachment rate

For a detached site ($\pi_{nm} = 0$), the probability to attach to a new microtubule in the time interval $dt$ is given by: $P_a = 1 - \exp(k_a\,dt)$. If an attachment event occurs, it can be to the left SPB with a probability $P_L$ such that:

\begin{equation}
  \label{eq:p_left}
  P_L =  1/2 + \beta \frac{N_n^{AL} - N_n^{BL}}{2(N_n^{AL} + N_n^{BL})}
\end{equation}

### Detachment rate

The detachment rate $k_d$ depends on the position of the attachment site with respect to the centromere\footnote{The following expression diverges when $ x_{nm}^A = x_n^A $, but this is only means the probability tends to 1. In the simulation code, a cut off value for $k_d$ is given.} :

\begin{equation}
  \label{eq:k_det}
  k_d = k_d^0 \frac{d_\alpha}{|(x_{nm}^A + x_{nm}^B)/2 - x_n^A|}
\end{equation}

## Length-dependent pulling force

To model the centering of chromosomes during metaphase, a length-dependent parameter has been added to the pulling force at each attachment site. Then the typical force-velocity relationship (published in Mary et al., (2015)) :

\begin{equation}
F_{kMT}= \pi F_k (1 - \frac{v}{V_k})\\
\end{equation}

becomes :

\begin{equation}
F_{kMT}= L_{dep} \pi F_k (1 - \frac{v}{V_k})\\
\end{equation}

The prefactor $L_{dep}$ adapts the force applied on the attachment site according to the distance $d_{site-pole}$ between this attachment site and the pole it is attached, such that :

\begin{equation}
L_{dep} = 1 + \alpha(d_{site-pole} - d_{mean})\\
\end{equation}

where $\alpha$ is a free parameter and $d_{mean}$ is the mean in vivo distance between kinetochores and the poles (i.e, approximately half the size of the spindle). Therefore, the stall force is unchanged when the attachment site is near the spindle center, while it is incresed (resp. decreased) when the attachment site is far from (resp. close to) the pole it is attached.
