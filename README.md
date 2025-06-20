# Derivative-Pricing

<a align='center'>
<img src='plot.png'/>
</a>

## Introduction
In computational finance and risk management, several numerical methods (e.g., finited differences, fourier methods, and Monte Carlo simulation) are commonly used for the valuation of financial derivatives.

The Black-Scholes formula is probably one of the most widely cited and used models in derivative pricing. Numerous variations and extensions of this formula are used to price many kinds of financial derivatives. However, the model is based on several assumptions. It assumes a specific form of movement for the derivative price, namely a Geometric Brownian Motion (GBM). It also assumes a conditional payment at maturity of the option and economic constraints, such as no-arbitrage. Several other derivate pricing models have similarly impractical model assumptions. Finance practitioners are well aware that these assumptions are violated in practice, and prices from these models are further adjusted using practitioner judgement.

Another aspect of the many traditional derivative pricing models is model calibration, which is typically done not by historical asset prices but by means of derivative prices (i.e., by matching the market prices of a heavily traded options to the derivative prices from the mathematical model). In the process of model calibration, thousands of derivative prices need to be determined in order to fit the parameters of the model, increasingly important in fianancial risk management, especially when we deal with real-time risk management (e.g., high frequency trading). However, due to the methodologies are discarded during model calibration of traditional derivative pricing models.

Machine learning can potentially be used to tackle these drawbacks related to impractical model assumptions and inefficient model calibration. Machine learning algorithms have the ability to tacke more nuances with very few theoretical assumptions and can be effectively used for derivative pricing, even in a world with frictions. With the advancements in hardware, we can train machine learning models on high performance CPUs, GPUs and other specialized hardware to achieve a speed increase of several orders of magnitude as compared to the traditional derivative pricing models.

Additionally, market data is plentiful, so it is possible to train a machine learning algorithm to learn the function that is collectively generation derivative prices in the market. Machine learning models can caputre subtle nonlinearities in the data these are no obtainable through other statistical approaches.

In this case study, we loot at derivative pricing from a machine learning standpoint and use a supervised regression-based model to price an option from simulated data. The main idea here is to come up with a machine learning framework for derivative pricing. Achieving a machine learning model with high accuracy would mean that we can leverage the efficient numerical calculation of machine learning for derivative pricing with fewer underlying model assumptions.

## Problem Definition

In the supervised regression framework we used for this case study, the predicted variable is the price of the option, and the predictor variables are the market data used as inputs to the Black-Scholes option pricing model.

The variables selected to estimate the market price of the option are stock price, strike price, time to expiration, volatility, interest rate and dividend yield. The predicted variable for this case study was generated using random inputs and feeding them into the well-known Black-Scholes model.

The price of a call option per the Black-Scholes option pricing model is defined in given equation.

<p align="center">
<img src='call option.png'/>
</p>

where: 
C = Call option price
S = Current price of the underlying asset
K = Strike price of the option
r = Risk-free interest rate
T = Time to expiration

To make the logic simpler, we define moneyness as M = K / S and look at the price in terms of per unit of current stock price.

Looking at the equation above, the parameters that feed into the Black-Scholes option pricing model are moneyness, risk-free rate, volatility, and time to maturity.

The parameter that plays the central role in derivative market is volatility, as it is directly related to the movement of the stock prices. With the increase in the volatility, the range of the share price movements becomes much wider than that of a low volatility stock.

In the options market, there isn't a single volatility used to price all the options. This volatility depends on the option moneyness. This behaviour is referred to as volatility smile/skew. We often derive the volatility from the price of the options existing in the market, and this volatility surface and use function, where volatility depends on the option moneyness and time to maturity to generate the option volatility surface.
