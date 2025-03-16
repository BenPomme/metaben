//+------------------------------------------------------------------+
//|                                    ML_Enhanced_EA.mq5            |
//|                                    Copyright 2025, MetaBen       |
//|                                                                  |
//| A simplified machine learning enhanced strategy for MetaTrader 5 |
//| Optimized for EURUSD with 1:500 leverage                        |
//+------------------------------------------------------------------+
#property copyright "2025, MetaBen"
#property link      ""
#property version   "1.00"
#property strict

// Include necessary files
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Arrays\ArrayObj.mqh>

// Input parameters - Machine Learning Features
input int      PredictionPeriod = 5;            // Forward prediction period
input double   PredictionThreshold = 0.55;      // Probability threshold for signals (0.5 to 1.0)
input bool     UseMultiTimeframe = true;        // Use multi-timeframe analysis

// Input parameters - Technical Indicators
input int      FastMA = 10;                     // Fast MA period
input int      SlowMA = 30;                     // Slow MA period
input int      BBPeriod = 20;                   // Bollinger Bands period
input double   BBDeviation = 2.0;               // Bollinger Bands deviation
input int      RSIPeriod = 14;                  // RSI period
input int      RSIOversold = 30;                // RSI oversold level
input int      RSIOverbought = 70;              // RSI overbought level
input int      ATRPeriod = 14;                  // ATR period

// Input parameters - Money Management
input double   RiskPercent = 2.0;               // Risk per trade (%)
input double   MaxLeverage = 500.0;             // Maximum leverage (1:500)
input double   MaxRiskPerTrade = 5.0;           // Maximum risk per trade (%)
input double   DailyProfitTarget = 1.0;         // Daily profit target (%)
input int      MaxTradesPerDay = 5;             // Maximum trades per day
input double   ATRMultiplierSL = 2.5;           // ATR multiplier for stop loss (increased)
input double   ATRMultiplierTP = 5.0;           // ATR multiplier for take profit (increased)
input int      MinStopDistancePips = 10;        // Minimum stop distance in pips

// Input parameters - Time Filters
input bool     UseTimeFilter = false;           // Use time filter
input string   TradingStartTime = "00:00";      // Trading start time
input string   TradingEndTime = "23:59";        // Trading end time
input bool     MondayFilter = false;            // Filter out Monday
input bool     FridayFilter = false;            // Filter out Friday

// Input parameters - Timeframes
input ENUM_TIMEFRAMES PrimaryTimeframe = PERIOD_H1;   // Primary timeframe
input ENUM_TIMEFRAMES SecondaryTimeframe = PERIOD_H4; // Secondary timeframe
input ENUM_TIMEFRAMES TertiaryTimeframe = PERIOD_D1;  // Tertiary timeframe

// Global variables
CTrade        Trade;
CSymbolInfo   SymbolInfo;
datetime      LastBarTime;
int           TodayTrades = 0;
datetime      TodayStartTime = 0;
double        DailyProfitPct = 0.0;
double        InitialBalance;
double        CurrentBalance;

// Indicator handles
int           FastMAHandle, SlowMAHandle;
int           RSIHandle;
int           BBHandle;
int           ATRHandle;

// Signal variables
bool          BuySignal = false;
bool          SellSignal = false;
bool          ExitLongSignal = false;
bool          ExitShortSignal = false;

// Feature storage
double        Features[][10];  // Store calculated features for ML prediction

// Near top of the file, add price consistency checks
string          g_LastPriceError = "";
bool            g_ForceTradingDisabled = false;
double          g_LastVerifiedBid = 0;
double          g_LastVerifiedAsk = 0;
datetime        lastPriceCheck = 0;  // Time of the last price check

// Add in the declaration section:
MqlTick current_tick;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize CSymbolInfo
   if(!SymbolInfo.Name(_Symbol))
   {
      Print("Failed to set symbol");
      return INIT_FAILED;
   }
   
   // Verify price data at startup
   g_ForceTradingDisabled = false;
   g_LastPriceError = "";
   
   // Get prices from multiple sources
   if(!SymbolInfo.RefreshRates())
   {
      Print("Failed to refresh rates");
      return INIT_FAILED;
   }
   
   double syminfo_bid = SymbolInfo.Bid();
   double syminfo_ask = SymbolInfo.Ask();
   double direct_bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double direct_ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   // Get recent close price
   double close[];
   ArraySetAsSeries(close, true);
   if(CopyClose(_Symbol, PrimaryTimeframe, 0, 1, close) != 1)
   {
      Print("Failed to copy close price at initialization");
      return INIT_FAILED;
   }
   
   // Check for price discrepancies at initialization
   if(MathAbs(syminfo_bid - direct_bid) > 0.0001 || 
      MathAbs(syminfo_ask - direct_ask) > 0.0001 ||
      MathAbs(direct_bid - close[0]) > 0.01)
   {
      string error_msg = StringFormat("STARTUP PRICE MISMATCH DETECTED! SymbolInfo: %.5f/%.5f, Direct: %.5f/%.5f, Close: %.5f",
                                     syminfo_bid, syminfo_ask, direct_bid, direct_ask, close[0]);
      g_LastPriceError = error_msg;
      Print(error_msg);
      
      // Warn about this issue but don't prevent initialization
      Print("WARNING: Price mismatch detected at initialization. The EA will use direct market prices.");
   }
   
   Print("STARTUP PRICES - SymbolInfo: Bid=", syminfo_bid, " Ask=", syminfo_ask, 
         "| Direct: Bid=", direct_bid, " Ask=", direct_ask,
         "| Recent Close=", close[0]);
   
   // Store verified prices
   g_LastVerifiedBid = direct_bid;
   g_LastVerifiedAsk = direct_ask;
   
   // Initialize Indicators
   FastMAHandle = iMA(_Symbol, PrimaryTimeframe, FastMA, 0, MODE_EMA, PRICE_CLOSE);
   SlowMAHandle = iMA(_Symbol, PrimaryTimeframe, SlowMA, 0, MODE_EMA, PRICE_CLOSE);
   RSIHandle = iRSI(_Symbol, PrimaryTimeframe, RSIPeriod, PRICE_CLOSE);
   BBHandle = iBands(_Symbol, PrimaryTimeframe, BBPeriod, 0, BBDeviation, PRICE_CLOSE);
   ATRHandle = iATR(_Symbol, PrimaryTimeframe, ATRPeriod);
   
   if(FastMAHandle == INVALID_HANDLE || SlowMAHandle == INVALID_HANDLE || 
      RSIHandle == INVALID_HANDLE || BBHandle == INVALID_HANDLE || ATRHandle == INVALID_HANDLE)
   {
      Print("Failed to create indicator handles");
      return INIT_FAILED;
   }
   
   // Initialize LastBarTime
   LastBarTime = iTime(_Symbol, PrimaryTimeframe, 0);
   
   // Initialize balance tracking
   InitialBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   CurrentBalance = InitialBalance;
   
   // Initialize daily tracking
   TodayStartTime = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
   ResetDailyStats();
   
   // Set trading parameters
   Trade.SetDeviationInPoints(10);
   Trade.SetTypeFilling(ORDER_FILLING_FOK);
   Trade.SetMarginMode();
   Trade.LogLevel(LOG_LEVEL_ERRORS);
   
   // Initialize feature array
   ArrayResize(Features, 100);
   
   Print("ML Enhanced EA initialized successfully");
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Release indicator handles
   IndicatorRelease(FastMAHandle);
   IndicatorRelease(SlowMAHandle);
   IndicatorRelease(RSIHandle);
   IndicatorRelease(BBHandle);
   IndicatorRelease(ATRHandle);
   
   Print("ML Enhanced EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Verify price data early to detect issues
   if(!VerifyPriceData())
   {
      Print("WARNING: Price data verification failed. Trading temporarily disabled.");
      return;
   }
   
   // Check for new bar
   datetime current_time = iTime(_Symbol, PrimaryTimeframe, 0);
   if(current_time == LastBarTime)
      return;
   
   // Update LastBarTime
   LastBarTime = current_time;
   
   Print("--- New bar detected at: ", TimeToString(current_time));
   
   // Check if we need to reset daily stats
   if(TimeCurrent() >= TodayStartTime + 86400) // 86400 seconds = 1 day
   {
      TodayStartTime = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
      ResetDailyStats();
   }
   
   // Update current balance
   CurrentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   
   // Calculate daily profit percentage
   DailyProfitPct = (CurrentBalance - InitialBalance) / InitialBalance * 100.0;
   
   // Check if daily profit target is reached
   if(DailyProfitPct >= DailyProfitTarget)
   {
      Print("Daily profit target reached: ", DoubleToString(DailyProfitPct, 2), "%");
      return;
   }
   
   // Check if maximum trades per day is reached
   if(TodayTrades >= MaxTradesPerDay)
   {
      Print("Maximum trades per day reached: ", TodayTrades);
      return;
   }
   
   // Check time filter
   if(UseTimeFilter && !IsTradeTimeAllowed())
   {
      Print("Trading not allowed at current time");
      return;
   }
   
   // Check day filter
   MqlDateTime time_struct;
   TimeCurrent(time_struct);
   if((MondayFilter && time_struct.day_of_week == 1) || 
      (FridayFilter && time_struct.day_of_week == 5))
   {
      Print("Trading not allowed on current day: ", time_struct.day_of_week);
      return;
   }
   
   // Process existing positions first
   if(PositionsTotal() > 0)
   {
      ManageOpenPositions();
   }
   else
   {
      // Calculate features and make prediction
      CalculateFeatures();
      MakePrediction();
      
      Print("Signal status - Buy: ", BuySignal ? "TRUE" : "FALSE", 
            ", Sell: ", SellSignal ? "TRUE" : "FALSE");
      
      if(BuySignal)
      {
         OpenBuyPosition();
      }
      else if(SellSignal)
      {
         OpenSellPosition();
      }
      else
      {
         Print("No valid signals generated");
      }
   }
}

//+------------------------------------------------------------------+
//| Calculate technical features for ML prediction                   |
//+------------------------------------------------------------------+
bool CalculateFeatures()
{
   // Verify we're using accurate price data before calculating features
   if(TimeCurrent() - lastPriceCheck > 5) // If prices haven't been checked in the last 5 seconds
   {
      if(!VerifyPriceData())
      {
         Print("WARNING: Price data verification failed in CalculateFeatures. Skipping calculations.");
         return false;
      }
   }
   
   // Number of bars to analyze
   int bars_required = SlowMA + 20;
   
   // Price data arrays
   double close[], open[], high[], low[];
   long volume[];  // Changed from double to long
   double fast_ma[], slow_ma[], rsi[], bb_upper[], bb_middle[], bb_lower[], atr[];
   
   // Set arrays as series
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(volume, true);
   ArraySetAsSeries(fast_ma, true);
   ArraySetAsSeries(slow_ma, true);
   ArraySetAsSeries(rsi, true);
   ArraySetAsSeries(bb_upper, true);
   ArraySetAsSeries(bb_middle, true);
   ArraySetAsSeries(bb_lower, true);
   ArraySetAsSeries(atr, true);
   
   // Copy price data
   if(CopyClose(_Symbol, PrimaryTimeframe, 0, bars_required, close) != bars_required ||
      CopyOpen(_Symbol, PrimaryTimeframe, 0, bars_required, open) != bars_required ||
      CopyHigh(_Symbol, PrimaryTimeframe, 0, bars_required, high) != bars_required ||
      CopyLow(_Symbol, PrimaryTimeframe, 0, bars_required, low) != bars_required ||
      CopyTickVolume(_Symbol, PrimaryTimeframe, 0, bars_required, volume) != bars_required)
   {
      Print("Error copying price data");
      return false;
   }
   
   // Copy indicator data
   if(CopyBuffer(FastMAHandle, 0, 0, bars_required, fast_ma) != bars_required ||
      CopyBuffer(SlowMAHandle, 0, 0, bars_required, slow_ma) != bars_required ||
      CopyBuffer(RSIHandle, 0, 0, bars_required, rsi) != bars_required ||
      CopyBuffer(BBHandle, 0, 0, bars_required, bb_upper) != bars_required ||
      CopyBuffer(BBHandle, 1, 0, bars_required, bb_middle) != bars_required ||
      CopyBuffer(BBHandle, 2, 0, bars_required, bb_lower) != bars_required ||
      CopyBuffer(ATRHandle, 0, 0, bars_required, atr) != bars_required)
   {
      Print("Error copying indicator data");
      return false;
   }
   
   // Calculate and store features for the current bar
   
   // Feature 1: Price relative to fast MA (normalized)
   Features[0][0] = (close[0] - fast_ma[0]) / fast_ma[0];
   
   // Feature 2: Price relative to slow MA (normalized)
   Features[0][1] = (close[0] - slow_ma[0]) / slow_ma[0];
   
   // Feature 3: Fast MA relative to slow MA (normalized)
   Features[0][2] = (fast_ma[0] - slow_ma[0]) / slow_ma[0];
   
   // Feature 4: RSI value (already normalized 0-100)
   Features[0][3] = rsi[0] / 100.0;
   
   // Feature 5: Price relative to Bollinger middle band (normalized)
   Features[0][4] = (close[0] - bb_middle[0]) / bb_middle[0];
   
   // Feature 6: Bollinger band width (normalized)
   double bb_width = (bb_upper[0] - bb_lower[0]) / bb_middle[0];
   Features[0][5] = bb_width;
   
   // Feature 7: Price volatility (ATR relative to price, normalized)
   Features[0][6] = atr[0] / close[0];
   
   // Feature 8: Volume change (normalized)
   double vol_change = volume[0] > 0 && volume[1] > 0 ? ((double)volume[0] - (double)volume[1]) / (double)volume[1] : 0;
   Features[0][7] = vol_change;
   
   // Feature 9: Price momentum (3-bar close change, normalized)
   double momentum = close[0] > 0 ? (close[0] - close[3]) / close[3] : 0;
   Features[0][8] = momentum;
   
   // Feature 10: Price range relative to ATR
   double bar_range = (high[0] - low[0]) / atr[0];
   Features[0][9] = bar_range;
   
   Print("Features calculated - Price/FastMA: ", Features[0][0], 
         ", RSI: ", Features[0][3] * 100, 
         ", BB Position: ", Features[0][4],
         ", ATR Ratio: ", Features[0][6]);
   
   return true;
}

//+------------------------------------------------------------------+
//| Make trading decision based on features                          |
//+------------------------------------------------------------------+
void MakePrediction()
{
   // Reset signals
   BuySignal = false;
   SellSignal = false;
   ExitLongSignal = false;
   ExitShortSignal = false;
   
   // Simplified ML model implemented directly in code
   // This is a rules-based approximation of our ML model
   
   double uptrend_score = 0;
   double downtrend_score = 0;
   
   // Rule 1: Price relative to moving averages
   if(Features[0][0] > 0 && Features[0][1] > 0) // Price above both MAs
      uptrend_score += 0.3;
   else if(Features[0][0] < 0 && Features[0][1] < 0) // Price below both MAs
      downtrend_score += 0.3;
   
   // Rule 2: Fast MA relative to slow MA
   if(Features[0][2] > 0) // Fast MA above slow MA
      uptrend_score += 0.2;
   else
      downtrend_score += 0.2;
   
   // Rule 3: RSI conditions
   if(Features[0][3] > 0.5 && Features[0][3] < 0.7) // RSI between 50-70
      uptrend_score += 0.15;
   else if(Features[0][3] < 0.5 && Features[0][3] > 0.3) // RSI between 30-50
      downtrend_score += 0.15;
   
   // Rule 4: Bollinger band position
   if(Features[0][4] > 0) // Price above middle band
      uptrend_score += 0.15;
   else
      downtrend_score += 0.15;
   
   // Rule 5: Momentum
   if(Features[0][8] > 0) // Positive momentum
      uptrend_score += 0.2;
   else
      downtrend_score += 0.2;
   
   // Apply threshold for signals
   if(uptrend_score >= PredictionThreshold && uptrend_score > downtrend_score)
   {
      BuySignal = true;
      Print("ML Buy signal generated with confidence: ", uptrend_score);
   }
   else if(downtrend_score >= PredictionThreshold && downtrend_score > uptrend_score)
   {
      SellSignal = true;
      Print("ML Sell signal generated with confidence: ", downtrend_score);
   }
   else
   {
      Print("No clear signal - Up score: ", uptrend_score, ", Down score: ", downtrend_score);
   }
   
   // Check multi-timeframe alignment if enabled
   if(UseMultiTimeframe && (BuySignal || SellSignal))
   {
      bool higher_tf_aligned = CheckHigherTimeframeAlignment();
      
      if(!higher_tf_aligned)
      {
         Print("Signal rejected due to higher timeframe misalignment");
         BuySignal = false;
         SellSignal = false;
      }
   }
   
   // Set exit signals
   SetExitSignals();
}

//+------------------------------------------------------------------+
//| Check if higher timeframes align with primary timeframe signal   |
//+------------------------------------------------------------------+
bool CheckHigherTimeframeAlignment()
{
   int tf_fast_ma = iMA(_Symbol, SecondaryTimeframe, FastMA, 0, MODE_EMA, PRICE_CLOSE);
   int tf_slow_ma = iMA(_Symbol, SecondaryTimeframe, SlowMA, 0, MODE_EMA, PRICE_CLOSE);
   
   if(tf_fast_ma == INVALID_HANDLE || tf_slow_ma == INVALID_HANDLE)
      return false;
   
   double tf_fast_ma_values[], tf_slow_ma_values[], tf_close[];
   ArraySetAsSeries(tf_fast_ma_values, true);
   ArraySetAsSeries(tf_slow_ma_values, true);
   ArraySetAsSeries(tf_close, true);
   
   if(CopyBuffer(tf_fast_ma, 0, 0, 2, tf_fast_ma_values) != 2 ||
      CopyBuffer(tf_slow_ma, 0, 0, 2, tf_slow_ma_values) != 2 ||
      CopyClose(_Symbol, SecondaryTimeframe, 0, 2, tf_close) != 2)
   {
      IndicatorRelease(tf_fast_ma);
      IndicatorRelease(tf_slow_ma);
      return false;
   }
   
   bool alignment = false;
   
   if(BuySignal)
   {
      // Check if higher timeframe supports buy signal
      bool h4_uptrend = tf_fast_ma_values[0] > tf_slow_ma_values[0];
      bool h4_price_above_ma = tf_close[0] > tf_fast_ma_values[0];
      
      alignment = h4_uptrend && h4_price_above_ma;
   }
   else if(SellSignal)
   {
      // Check if higher timeframe supports sell signal
      bool h4_downtrend = tf_fast_ma_values[0] < tf_slow_ma_values[0];
      bool h4_price_below_ma = tf_close[0] < tf_fast_ma_values[0];
      
      alignment = h4_downtrend && h4_price_below_ma;
   }
   
   IndicatorRelease(tf_fast_ma);
   IndicatorRelease(tf_slow_ma);
   
   return alignment;
}

//+------------------------------------------------------------------+
//| Set exit signals for open positions                              |
//+------------------------------------------------------------------+
void SetExitSignals()
{
   double close[], rsi[], bb_upper[], bb_lower[];
   double fast_ma[], slow_ma[], macd_main[], macd_signal[];
   
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(rsi, true);
   ArraySetAsSeries(bb_upper, true);
   ArraySetAsSeries(bb_lower, true);
   ArraySetAsSeries(fast_ma, true);
   ArraySetAsSeries(slow_ma, true);
   
   // Copy data (only need a few bars for exit signals)
   if(CopyClose(_Symbol, PrimaryTimeframe, 0, 3, close) != 3 ||
      CopyBuffer(RSIHandle, 0, 0, 3, rsi) != 3 ||
      CopyBuffer(BBHandle, 0, 0, 3, bb_upper) != 3 ||
      CopyBuffer(BBHandle, 2, 0, 3, bb_lower) != 3 ||
      CopyBuffer(FastMAHandle, 0, 0, 3, fast_ma) != 3 ||
      CopyBuffer(SlowMAHandle, 0, 0, 3, slow_ma) != 3)
   {
      return;
   }
   
   // Exit long position conditions
   ExitLongSignal = (
      rsi[0] > RSIOverbought ||              // RSI overbought
      close[0] > bb_upper[0] ||              // Price above upper BB
      (fast_ma[0] < slow_ma[0] && fast_ma[1] > slow_ma[1]) // MA crossover down
   );
   
   // Exit short position conditions
   ExitShortSignal = (
      rsi[0] < RSIOversold ||                // RSI oversold
      close[0] < bb_lower[0] ||              // Price below lower BB
      (fast_ma[0] > slow_ma[0] && fast_ma[1] < slow_ma[1]) // MA crossover up
   );
}

//+------------------------------------------------------------------+
//| Open a buy position                                              |
//+------------------------------------------------------------------+
void OpenBuyPosition()
{
   // First check if trading is disabled due to price issues
   if(g_ForceTradingDisabled)
   {
      Print("TRADING DISABLED due to price data inconsistency. Please check logs and restart EA.");
      return;
   }

   double close[], atr[];
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(atr, true);
   
   // Copy recent price data for our own use
   if(CopyClose(_Symbol, PrimaryTimeframe, 0, 3, close) != 3 ||
      CopyBuffer(ATRHandle, 0, 0, 1, atr) != 1)
   {
      Print("Failed to copy price/ATR data");
      return;
   }
   
   // Get the latest tick data with all prices at once
   // This is the recommended approach in MQL5 documentation
   if(!SymbolInfoTick(_Symbol, current_tick))
   {
      Print("Failed to get current tick data. Error: ", GetLastError());
      return;
   }
   
   // Compare the tick data with the close price
   double current_close = close[0];
   Print("PRICE CHECK - Current Tick: Bid=", current_tick.bid, " Ask=", current_tick.ask, 
         " | Recent Close=", current_close, 
         " | Time diff: ", TimeCurrent() - current_tick.time, " seconds");
   
   // Check for severe price discrepancy
   if(MathAbs(current_tick.bid - current_close) > 0.01) // 100 pips difference
   {
      string error_msg = StringFormat("CRITICAL PRICE MISMATCH DETECTED! Tick: %.5f/%.5f, Close: %.5f",
                                    current_tick.bid, current_tick.ask, current_close);
      g_LastPriceError = error_msg;
      Print(error_msg);
      
      g_ForceTradingDisabled = true;
      Print("SEVERE PRICE MISMATCH - Trading has been disabled for safety. Please restart EA.");
      return;
   }
   
   // Get broker-specific information
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   long stop_level_points = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   
   // Get entry price from the tick data
   double entry_price = current_tick.ask;
   
   Print("Symbol info - Point: ", point, ", Digits: ", digits, ", Stop level: ", stop_level_points, " points");
   Print("ORDER PLACEMENT USING PRICES - Entry: ", entry_price, ", Bid: ", current_tick.bid, ", Ask: ", current_tick.ask);
   
   // Calculate minimum stop distance required by broker (in price)
   double min_stop_distance = stop_level_points * point;
   if(min_stop_distance < MinStopDistancePips * point)
      min_stop_distance = MinStopDistancePips * point;
   
   // Calculate stop levels using ATR
   double stop_loss = NormalizeDouble(entry_price - (atr[0] * ATRMultiplierSL), digits);
   double take_profit = NormalizeDouble(entry_price + (atr[0] * ATRMultiplierTP), digits);
   
   // Ensure minimum distance for stop loss
   if(MathAbs(entry_price - stop_loss) < min_stop_distance)
   {
      stop_loss = NormalizeDouble(entry_price - min_stop_distance, digits);
      Print("Stop loss adjusted to meet minimum distance requirement: ", stop_loss);
   }
   
   // Ensure minimum distance for take profit
   if(MathAbs(take_profit - entry_price) < min_stop_distance)
   {
      take_profit = NormalizeDouble(entry_price + min_stop_distance, digits);
      Print("Take profit adjusted to meet minimum distance requirement: ", take_profit);
   }
   
   // Calculate position size based on risk
   double lot_size = CalculatePositionSize(entry_price, stop_loss);
   
   if(lot_size == 0)
   {
      Print("Invalid lot size calculated");
      return;
   }
   
   // Normalize lot size according to broker requirements
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   lot_size = MathMax(min_lot, MathMin(max_lot, lot_size));
   lot_size = NormalizeDouble(lot_size / lot_step, 0) * lot_step;
   
   // Print detailed information about the order we're attempting
   Print("ORDER DETAILS - Buy, Lot size: ", lot_size, 
         ", Entry: ", entry_price, 
         ", SL: ", stop_loss, " (distance: ", NormalizeDouble((entry_price - stop_loss)/point, 1), " points)", 
         ", TP: ", take_profit, " (distance: ", NormalizeDouble((take_profit - entry_price)/point, 1), " points)");
   
   // Clear any previous errors
   ResetLastError();
   
   // Attempt to place market order
   if(Trade.Buy(lot_size, _Symbol, 0, stop_loss, take_profit, "ML Enhanced"))
   {
      Print("Buy position opened: ", lot_size, " lots at ", entry_price, 
            ", SL: ", stop_loss, ", TP: ", take_profit);
      TodayTrades++;
   }
   else
   {
      int error_code = GetLastError();
      Print("Failed to open buy position. Error: ", error_code, " - ", ErrorDescription(error_code));
      
      // Try alternative approach with pending order if market order fails
      if(error_code == 4756 || error_code == 130) // Invalid stops or invalid price
      {
         // For pending orders, place the entry a bit further away
         double pending_price = NormalizeDouble(entry_price + (20 * point), digits);
         
         Print("Attempting alternative approach with pending order...");
         Print("Pending order details - Type: Buy Stop, Price: ", pending_price, 
               ", SL: ", stop_loss, ", TP: ", take_profit, 
               " (Current prices: Bid: ", current_tick.bid, ", Ask: ", current_tick.ask, ")");
         
         ResetLastError();
         if(Trade.BuyStop(lot_size, pending_price, _Symbol, 
                        stop_loss, take_profit, ORDER_TIME_GTC, 0, "ML Enhanced"))
         {
            Print("Buy Stop order placed at: ", pending_price);
            TodayTrades++;
         }
         else
         {
            error_code = GetLastError();
            Print("Failed to place Buy Stop order. Error: ", error_code, " - ", ErrorDescription(error_code));
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Open a sell position                                             |
//+------------------------------------------------------------------+
void OpenSellPosition()
{
   // First check if trading is disabled due to price issues
   if(g_ForceTradingDisabled)
   {
      Print("TRADING DISABLED due to price data inconsistency. Please check logs and restart EA.");
      return;
   }

   double close[], atr[];
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(atr, true);
   
   // Copy recent price data for our own use
   if(CopyClose(_Symbol, PrimaryTimeframe, 0, 3, close) != 3 ||
      CopyBuffer(ATRHandle, 0, 0, 1, atr) != 1)
   {
      Print("Failed to copy price/ATR data");
      return;
   }
   
   // Get the latest tick data with all prices at once
   // This is the recommended approach in MQL5 documentation
   if(!SymbolInfoTick(_Symbol, current_tick))
   {
      Print("Failed to get current tick data. Error: ", GetLastError());
      return;
   }
   
   // Compare the tick data with the close price
   double current_close = close[0];
   Print("PRICE CHECK - Current Tick: Bid=", current_tick.bid, " Ask=", current_tick.ask, 
         " | Recent Close=", current_close, 
         " | Time diff: ", TimeCurrent() - current_tick.time, " seconds");
   
   // Check for severe price discrepancy
   if(MathAbs(current_tick.bid - current_close) > 0.01) // 100 pips difference
   {
      string error_msg = StringFormat("CRITICAL PRICE MISMATCH DETECTED! Tick: %.5f/%.5f, Close: %.5f",
                                    current_tick.bid, current_tick.ask, current_close);
      g_LastPriceError = error_msg;
      Print(error_msg);
      
      g_ForceTradingDisabled = true;
      Print("SEVERE PRICE MISMATCH - Trading has been disabled for safety. Please restart EA.");
      return;
   }
   
   // Get broker-specific information
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   long stop_level_points = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   
   // Get entry price from the tick data
   double entry_price = current_tick.bid;
   
   Print("Symbol info - Point: ", point, ", Digits: ", digits, ", Stop level: ", stop_level_points, " points");
   Print("ORDER PLACEMENT USING PRICES - Entry: ", entry_price, ", Bid: ", current_tick.bid, ", Ask: ", current_tick.ask);
   
   // Calculate minimum stop distance required by broker (in price)
   double min_stop_distance = stop_level_points * point;
   if(min_stop_distance < MinStopDistancePips * point)
      min_stop_distance = MinStopDistancePips * point;
   
   // Calculate stop levels using ATR
   double stop_loss = NormalizeDouble(entry_price + (atr[0] * ATRMultiplierSL), digits);
   double take_profit = NormalizeDouble(entry_price - (atr[0] * ATRMultiplierTP), digits);
   
   // Ensure minimum distance for stop loss
   if(MathAbs(stop_loss - entry_price) < min_stop_distance)
   {
      stop_loss = NormalizeDouble(entry_price + min_stop_distance, digits);
      Print("Stop loss adjusted to meet minimum distance requirement: ", stop_loss);
   }
   
   // Ensure minimum distance for take profit
   if(MathAbs(entry_price - take_profit) < min_stop_distance)
   {
      take_profit = NormalizeDouble(entry_price - min_stop_distance, digits);
      Print("Take profit adjusted to meet minimum distance requirement: ", take_profit);
   }
   
   // Calculate position size based on risk
   double lot_size = CalculatePositionSize(entry_price, stop_loss);
   
   if(lot_size == 0)
   {
      Print("Invalid lot size calculated");
      return;
   }
   
   // Normalize lot size according to broker requirements
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   lot_size = MathMax(min_lot, MathMin(max_lot, lot_size));
   lot_size = NormalizeDouble(lot_size / lot_step, 0) * lot_step;
   
   // Print detailed information about the order we're attempting
   Print("ORDER DETAILS - Sell, Lot size: ", lot_size, 
         ", Entry: ", entry_price, 
         ", SL: ", stop_loss, " (distance: ", NormalizeDouble((stop_loss - entry_price)/point, 1), " points)", 
         ", TP: ", take_profit, " (distance: ", NormalizeDouble((entry_price - take_profit)/point, 1), " points)");
   
   // Clear any previous errors
   ResetLastError();
   
   // Attempt to place market order
   if(Trade.Sell(lot_size, _Symbol, 0, stop_loss, take_profit, "ML Enhanced"))
   {
      Print("Sell position opened: ", lot_size, " lots at ", entry_price, 
            ", SL: ", stop_loss, ", TP: ", take_profit);
      TodayTrades++;
   }
   else
   {
      int error_code = GetLastError();
      Print("Failed to open sell position. Error: ", error_code, " - ", ErrorDescription(error_code));
      
      // Try alternative approach with pending order if market order fails
      if(error_code == 4756 || error_code == 130) // Invalid stops or invalid price
      {
         // For pending orders, place the entry a bit further away
         double pending_price = NormalizeDouble(entry_price - (20 * point), digits);
         
         Print("Attempting alternative approach with pending order...");
         Print("Pending order details - Type: Sell Stop, Price: ", pending_price, 
               ", SL: ", stop_loss, ", TP: ", take_profit, 
               " (Current prices: Bid: ", current_tick.bid, ", Ask: ", current_tick.ask, ")");
         
         ResetLastError();
         if(Trade.SellStop(lot_size, pending_price, _Symbol, 
                        stop_loss, take_profit, ORDER_TIME_GTC, 0, "ML Enhanced"))
         {
            Print("Sell Stop order placed at: ", pending_price);
            TodayTrades++;
         }
         else
         {
            error_code = GetLastError();
            Print("Failed to place Sell Stop order. Error: ", error_code, " - ", ErrorDescription(error_code));
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check if stop loss and take profit levels are valid              |
//+------------------------------------------------------------------+
bool CheckStopLevels(double entry_price, double stop_loss, double take_profit)
{
   // Get minimum stop level from the broker (in points)
   long stop_level = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double min_dist = stop_level * point;
   
   double ask = SymbolInfo.Ask();
   double bid = SymbolInfo.Bid();
   double dist_to_sl, dist_to_tp;
   
   // For buy order
   if(entry_price >= bid)
   {
      dist_to_sl = entry_price - stop_loss;
      dist_to_tp = take_profit - entry_price;
   }
   // For sell order
   else
   {
      dist_to_sl = stop_loss - entry_price;
      dist_to_tp = entry_price - take_profit;
   }
   
   // Check if distances are adequate
   bool result = true;
   
   if(stop_level > 0)
   {
      if(dist_to_sl < min_dist)
      {
         Print("Stop Loss is too close to entry price! Min allowed: ", min_dist, " points, Actual: ", dist_to_sl / point, " points");
         result = false;
      }
      
      if(dist_to_tp < min_dist)
      {
         Print("Take Profit is too close to entry price! Min allowed: ", min_dist, " points, Actual: ", dist_to_tp / point, " points");
         result = false;
      }
   }
   
   // Check if prices are normalized correctly
   if(MathAbs(MathRound(stop_loss / point) - (stop_loss / point)) > 0.00000001)
   {
      Print("Stop Loss price is not properly normalized!");
      result = false;
   }
   
   if(MathAbs(MathRound(take_profit / point) - (take_profit / point)) > 0.00000001)
   {
      Print("Take Profit price is not properly normalized!");
      result = false;
   }
   
   return result;
}

//+------------------------------------------------------------------+
//| Calculate position size based on risk percentage                 |
//+------------------------------------------------------------------+
double CalculatePositionSize(double entry_price, double stop_loss)
{
   if(MathAbs(entry_price - stop_loss) < _Point)
   {
      Print("Error: Stop loss too close to entry price");
      return 0;
   }
   
   // Cap risk percentage
   double risk_percent = MathMin(RiskPercent, MaxRiskPerTrade);
   
   // Calculate risk amount
   double risk_amount = CurrentBalance * (risk_percent / 100.0);
   
   // Calculate stop loss distance
   double stop_distance = MathAbs(entry_price - stop_loss);
   
   // Calculate position size
   double contract_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   
   // Calculate pip value in account currency
   double pip_value = tick_value * (stop_distance / tick_size);
   
   double standard_lot_size = risk_amount / pip_value;
   
   // Check if position size exceeds leverage limits
   double margin_required = SymbolInfoDouble(_Symbol, SYMBOL_MARGIN_INITIAL) * standard_lot_size;
   double max_position_size = (AccountInfoDouble(ACCOUNT_MARGIN_FREE) * MaxLeverage) / margin_required;
   
   // Cap position size to leverage limit
   double position_size = MathMin(standard_lot_size, max_position_size);
   
   Print("Position size calculation - Risk %: ", risk_percent,
         ", Risk Amount: $", risk_amount,
         ", Stop Distance: ", stop_distance,
         ", Calculated Size: ", standard_lot_size,
         ", Final Size: ", position_size);
   
   return position_size;
}

//+------------------------------------------------------------------+
//| Manage open positions                                            |
//+------------------------------------------------------------------+
void ManageOpenPositions()
{
   // Check for each position
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0)
         continue;
      
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      
      if(PositionGetInteger(POSITION_MAGIC) != Trade.RequestMagic())
         continue;
      
      ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      
      if(pos_type == POSITION_TYPE_BUY && ExitLongSignal)
      {
         Print("Exit signal for buy position detected");
         ClosePosition(ticket);
      }
      else if(pos_type == POSITION_TYPE_SELL && ExitShortSignal)
      {
         Print("Exit signal for sell position detected");
         ClosePosition(ticket);
      }
   }
}

//+------------------------------------------------------------------+
//| Close a position by ticket                                       |
//+------------------------------------------------------------------+
void ClosePosition(ulong ticket)
{
   if(PositionSelectByTicket(ticket))
   {
      double volume = PositionGetDouble(POSITION_VOLUME);
      ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      
      if(Trade.PositionClose(ticket))
      {
         Print("Position closed: Ticket #", ticket, ", Type: ", 
               (pos_type == POSITION_TYPE_BUY ? "Buy" : "Sell"), 
               ", Volume: ", volume);
      }
      else
      {
         Print("Failed to close position. Error: ", GetLastError());
      }
   }
}

//+------------------------------------------------------------------+
//| Check if trading time is allowed                                 |
//+------------------------------------------------------------------+
bool IsTradeTimeAllowed()
{
   // Parse trading time strings
   int start_hour, start_minute, end_hour, end_minute;
   
   string start_parts[], end_parts[];
   StringSplit(TradingStartTime, ':', start_parts);
   StringSplit(TradingEndTime, ':', end_parts);
   
   if(ArraySize(start_parts) != 2 || ArraySize(end_parts) != 2)
   {
      Print("Invalid time format");
      return false;
   }
   
   start_hour = (int)StringToInteger(start_parts[0]);
   start_minute = (int)StringToInteger(start_parts[1]);
   end_hour = (int)StringToInteger(end_parts[0]);
   end_minute = (int)StringToInteger(end_parts[1]);
   
   // Get current time
   MqlDateTime time_struct;
   TimeCurrent(time_struct);
   
   // Calculate minutes from midnight
   int current_minutes = time_struct.hour * 60 + time_struct.min;
   int start_minutes = start_hour * 60 + start_minute;
   int end_minutes = end_hour * 60 + end_minute;
   
   // Check if current time is within allowed range
   if(start_minutes <= end_minutes)
   {
      // Normal case (e.g., 8:00 to 20:00)
      return (current_minutes >= start_minutes && current_minutes <= end_minutes);
   }
   else
   {
      // Overnight case (e.g., 20:00 to 8:00)
      return (current_minutes >= start_minutes || current_minutes <= end_minutes);
   }
}

//+------------------------------------------------------------------+
//| Reset daily statistics                                           |
//+------------------------------------------------------------------+
void ResetDailyStats()
{
   TodayTrades = 0;
   DailyProfitPct = 0.0;
   InitialBalance = AccountInfoDouble(ACCOUNT_BALANCE);
}

//+------------------------------------------------------------------+
//| Get text description for error code                              |
//+------------------------------------------------------------------+
string ErrorDescription(int error_code)
{
   string error_string;
   
   switch(error_code)
   {
      case 0:     error_string = "No error";                         break;
      case 4756:  error_string = "Invalid stops";                    break;
      case 130:   error_string = "Invalid price";                    break;
      case 4052:  error_string = "Invalid trade volume";             break;
      case 4063:  error_string = "Maximum orders reached";           break;
      case 4109:  error_string = "Insufficient margin";              break;
      case 4110:  error_string = "Order locked";                     break;
      default:    error_string = "Unknown error";
   }
   
   return error_string;
}

//+------------------------------------------------------------------+
//| Add this function to verify price data                          |
//+------------------------------------------------------------------+
bool VerifyPriceData()
{
   // Force refresh rates
   if(!SymbolInfo.RefreshRates())
   {
      Print("Failed to refresh rates");
      return false;
   }
   
   // Get prices from different sources
   double symbol_info_bid = SymbolInfo.Bid();
   double symbol_info_ask = SymbolInfo.Ask();
   double direct_bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double direct_ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   // Check for price mismatch
   bool price_mismatch = false;
   if(MathAbs(symbol_info_bid - direct_bid) > 0.0001)
   {
      Print("Price mismatch! SymbolInfo.Bid(): ", symbol_info_bid, " vs SYMBOL_BID: ", direct_bid);
      price_mismatch = true;
   }
   
   if(MathAbs(symbol_info_ask - direct_ask) > 0.0001)
   {
      Print("Price mismatch! SymbolInfo.Ask(): ", symbol_info_ask, " vs SYMBOL_ASK: ", direct_ask);
      price_mismatch = true;
   }
   
   // Store verified prices
   g_LastVerifiedBid = direct_bid;
   g_LastVerifiedAsk = direct_ask;
   lastPriceCheck = TimeCurrent();
   
   return !price_mismatch;
} 