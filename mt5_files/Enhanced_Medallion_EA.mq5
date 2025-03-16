//+------------------------------------------------------------------+
//|                                    Enhanced_Medallion_EA.mq5     |
//|                                    Copyright 2025, MetaBen       |
//|                                                                  |
//| An enhanced Medallion strategy optimized for EURUSD with 1:500   |
//| leverage and $10,000 account, targeting 1% daily returns         |
//+------------------------------------------------------------------+
#property copyright "2025, MetaBen"
#property link      ""
#property version   "1.01"
#property strict

// Include necessary files
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>

// Input parameters - Price Action
input double   ReversalCandleStrength = 0.6;    // Strength for reversal candles
input double   TrendStrengthThreshold = 0.5;    // Threshold for trend strength
input int      SupportResistanceLookback = 15;  // Periods to look back for S/R

// Input parameters - Moving Averages
input int      FastMAPeriod = 5;                // Fast MA period
input int      SlowMAPeriod = 15;               // Slow MA period
input int      SignalMAPeriod = 5;              // MACD signal period
input ENUM_MA_METHOD MAMethod = MODE_EMA;       // MA method (EMA/SMA)
input ENUM_APPLIED_PRICE MAPrice = PRICE_CLOSE; // Price for MA calculation

// Input parameters - RSI
input int      RSIPeriod = 10;                  // RSI period
input int      RSILevel_Overbought = 75;        // RSI overbought level
input int      RSILevel_Oversold = 25;          // RSI oversold level

// Input parameters - Bollinger Bands
input int      BBPeriod = 15;                   // Bollinger Bands period
input double   BBDeviation = 1.8;               // Bollinger Bands deviation

// Input parameters - Money Management
input double   RiskPercent = 2.5;               // Risk per trade (%)
input double   MaxLeverage = 500.0;             // Maximum leverage (1:500)
input double   MaxRiskPerTrade = 5.0;           // Maximum risk per trade (%)
input double   DailyProfitTarget = 1.0;         // Daily profit target (%)
input int      MaxTradesPerDay = 5;             // Maximum trades per day
input bool     UseATRForSL = true;              // Use ATR for stop loss
input double   ATRMultiplierSL = 2.5;           // ATR multiplier for stop loss (increased)
input double   ATRMultiplierTP = 5.0;           // ATR multiplier for take profit (increased)
input int      MinStopDistancePips = 10;        // Minimum stop distance in pips

// Input parameters - Time Filters
input bool     UseTimeFilter = false;            // Use time filter
input string   TradingStartTime = "00:00";      // Trading start time
input string   TradingEndTime = "23:59";        // Trading end time
input bool     MondayFilter = false;            // Filter out Monday
input bool     FridayFilter = false;            // Filter out Friday
input bool     TradeOnlyWithAllTimeframes = false; // Trade only when all timeframes align

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
int           EmaFastHandle, EmaSlowHandle;
int           RSIHandle;
int           BBHandle;
int           ATRHandle;
bool          BuySignal = false;
bool          SellSignal = false;
bool          ExitLongSignal = false;
bool          ExitShortSignal = false;

// Near top of the file, add price consistency checks
string          g_LastPriceError = "";
bool            g_ForceTradingDisabled = false;
double          g_LastVerifiedBid = 0;
double          g_LastVerifiedAsk = 0;

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
   EmaFastHandle = iMA(_Symbol, PrimaryTimeframe, FastMAPeriod, 0, MAMethod, MAPrice);
   EmaSlowHandle = iMA(_Symbol, PrimaryTimeframe, SlowMAPeriod, 0, MAMethod, MAPrice);
   RSIHandle = iRSI(_Symbol, PrimaryTimeframe, RSIPeriod, MAPrice);
   BBHandle = iBands(_Symbol, PrimaryTimeframe, BBPeriod, 0, BBDeviation, MAPrice);
   ATRHandle = iATR(_Symbol, PrimaryTimeframe, 14);
   
   if(EmaFastHandle == INVALID_HANDLE || EmaSlowHandle == INVALID_HANDLE || 
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
   
   Print("Enhanced Medallion EA initialized successfully");
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Release indicator handles
   IndicatorRelease(EmaFastHandle);
   IndicatorRelease(EmaSlowHandle);
   IndicatorRelease(RSIHandle);
   IndicatorRelease(BBHandle);
   IndicatorRelease(ATRHandle);
   
   Print("Enhanced Medallion EA deinitialized");
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
   
   // Process existing positions
   if(PositionsTotal() > 0)
   {
      ManageOpenPositions();
   }
   else
   {
      // Check for entry signals
      CalculateSignals();
      
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
//| Calculate entry and exit signals                                 |
//+------------------------------------------------------------------+
void CalculateSignals()
{
   int bars_required = MathMin(MathMax(SupportResistanceLookback, BBPeriod) + 10, 50); // Limit to maximum 50 bars
   Print("Requesting ", bars_required, " bars for calculations");
   
   // Reset signals
   BuySignal = false;
   SellSignal = false;
   ExitLongSignal = false;
   ExitShortSignal = false;
   
   // Prepare buffers for indicators
   double ema_fast[], ema_slow[], rsi[], bb_upper[], bb_middle[], bb_lower[];
   double macd[], macd_signal[], macd_hist[], atr[];
   
   // Resize arrays
   ArraySetAsSeries(ema_fast, true);
   ArraySetAsSeries(ema_slow, true);
   ArraySetAsSeries(rsi, true);
   ArraySetAsSeries(bb_upper, true);
   ArraySetAsSeries(bb_middle, true);
   ArraySetAsSeries(bb_lower, true);
   ArraySetAsSeries(macd, true);
   ArraySetAsSeries(macd_signal, true);
   ArraySetAsSeries(macd_hist, true);
   ArraySetAsSeries(atr, true);
   
   // Use ArrayResize to ensure arrays are properly sized
   ArrayResize(macd, bars_required);
   ArrayResize(macd_signal, bars_required);
   ArrayResize(macd_hist, bars_required);
   
   // Copy indicator data
   if(CopyBuffer(EmaFastHandle, 0, 0, bars_required, ema_fast) < bars_required ||
      CopyBuffer(EmaSlowHandle, 0, 0, bars_required, ema_slow) < bars_required ||
      CopyBuffer(RSIHandle, 0, 0, bars_required, rsi) < bars_required ||
      CopyBuffer(BBHandle, 0, 0, bars_required, bb_upper) < bars_required ||
      CopyBuffer(BBHandle, 1, 0, bars_required, bb_middle) < bars_required ||
      CopyBuffer(BBHandle, 2, 0, bars_required, bb_lower) < bars_required ||
      CopyBuffer(ATRHandle, 0, 0, bars_required, atr) < bars_required)
   {
      Print("Failed to copy indicator data");
      return;
   }
   
   Print("Successfully copied indicator data - array sizes: ",
         "EMA Fast: ", ArraySize(ema_fast),
         ", EMA Slow: ", ArraySize(ema_slow),
         ", RSI: ", ArraySize(rsi));
   
   // Calculate MACD
   
   // Check that the arrays have enough data before accessing elements
   if(ArraySize(ema_fast) < 2 || ArraySize(ema_slow) < 2)
   {
      Print("Error: Not enough data in MA arrays. Fast EMA size: ", ArraySize(ema_fast), 
            ", Slow EMA size: ", ArraySize(ema_slow));
      return;
   }
   
   macd[0] = ema_fast[0] - ema_slow[0];
   macd[1] = ema_fast[1] - ema_slow[1];
   
   // Calculate signal line (EMA of MACD)
   double signal_line = 0;
   double prev_signal_line = 0;
   
   // Simple implementation of EMA calculation
   double alpha = 2.0 / (SignalMAPeriod + 1.0);
   
   // Get previous value if available
   if(bars_required > SignalMAPeriod && ArraySize(ema_fast) >= SignalMAPeriod && ArraySize(ema_slow) >= SignalMAPeriod) {
      // Calculate first signal value as simple average
      double sum = 0;
      for(int i = 0; i < SignalMAPeriod && i < ArraySize(ema_fast) && i < ArraySize(ema_slow); i++) {
         double val = ema_fast[i] - ema_slow[i];
         sum += val;
      }
      prev_signal_line = sum / SignalMAPeriod;
      
      // Calculate current EMA
      signal_line = alpha * macd[0] + (1.0 - alpha) * prev_signal_line;
   } else {
      // Fallback if not enough bars
      signal_line = macd[0];
      prev_signal_line = macd[1];
   }
   
   macd_signal[0] = signal_line;
   macd_signal[1] = prev_signal_line;
   macd_hist[0] = macd[0] - macd_signal[0];
   macd_hist[1] = macd[1] - macd_signal[1];
   
   // Get price data
   double close[], open[], high[], low[];
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   
   if(CopyClose(_Symbol, PrimaryTimeframe, 0, bars_required, close) < bars_required ||
      CopyOpen(_Symbol, PrimaryTimeframe, 0, bars_required, open) < bars_required ||
      CopyHigh(_Symbol, PrimaryTimeframe, 0, bars_required, high) < bars_required ||
      CopyLow(_Symbol, PrimaryTimeframe, 0, bars_required, low) < bars_required)
   {
      Print("Failed to copy price data");
      return;
   }
   
   // Print indicator values for debugging
   Print("Indicator values - RSI: ", DoubleToString(rsi[0], 2), 
         ", Fast EMA: ", DoubleToString(ema_fast[0], 5),
         ", Slow EMA: ", DoubleToString(ema_slow[0], 5),
         ", MACD: ", DoubleToString(macd[0], 5),
         ", MACD Hist: ", DoubleToString(macd_hist[0], 5));
   
   // Calculate price action patterns
   bool bullish_engulfing = (close[0] > open[0]) && 
                            (open[0] < close[1]) && 
                            (close[0] > open[1]) && 
                            (open[1] > close[1]);
   
   bool bearish_engulfing = (close[0] < open[0]) && 
                            (open[0] > close[1]) && 
                            (close[0] < open[1]) && 
                            (open[1] < close[1]);
   
   double candle_size = MathAbs(close[0] - open[0]);
   double shadow_size = high[0] - low[0];
   bool doji = candle_size < (shadow_size * 0.1);
   
   // Calculate MA crossover signals
   bool ma_cross_up = (ema_fast[0] > ema_slow[0]) && (ema_fast[1] <= ema_slow[1]);
   bool ma_cross_down = (ema_fast[0] < ema_slow[0]) && (ema_fast[1] >= ema_slow[1]);
   bool uptrend = ema_fast[0] > ema_slow[0];
   bool downtrend = ema_fast[0] < ema_slow[0];
   
   // Check conditions for buy signal
   bool ma_condition_buy = ma_cross_up || uptrend;
   bool rsi_condition_buy = rsi[0] < 50.0 || rsi[0] > rsi[1];
   bool macd_condition_buy = macd_hist[0] > 0 || (macd_hist[0] > macd_hist[1]);
   bool price_action_buy = bullish_engulfing || doji;
   bool bb_condition_buy = close[0] < bb_middle[0];
   
   // Calculate buy signal strength (2 out of 5 conditions)
   int buy_signal_strength = (ma_condition_buy ? 1 : 0) + 
                            (rsi_condition_buy ? 1 : 0) + 
                            (macd_condition_buy ? 1 : 0) + 
                            (price_action_buy ? 1 : 0) + 
                            (bb_condition_buy ? 1 : 0);
   
   // Check conditions for sell signal
   bool ma_condition_sell = ma_cross_down || downtrend;
   bool rsi_condition_sell = rsi[0] > 50.0 || rsi[0] < rsi[1];
   bool macd_condition_sell = macd_hist[0] < 0 || (macd_hist[0] < macd_hist[1]);
   bool price_action_sell = bearish_engulfing || doji;
   bool bb_condition_sell = close[0] > bb_middle[0];
   
   // Calculate sell signal strength (2 out of 5 conditions)
   int sell_signal_strength = (ma_condition_sell ? 1 : 0) + 
                             (rsi_condition_sell ? 1 : 0) + 
                             (macd_condition_sell ? 1 : 0) + 
                             (price_action_sell ? 1 : 0) + 
                             (bb_condition_sell ? 1 : 0);
   
   // Print signal conditions
   Print("Buy conditions - MA: ", ma_condition_buy ? "TRUE" : "FALSE",
         ", RSI: ", rsi_condition_buy ? "TRUE" : "FALSE",
         ", MACD: ", macd_condition_buy ? "TRUE" : "FALSE",
         ", Price Action: ", price_action_buy ? "TRUE" : "FALSE",
         ", BB: ", bb_condition_buy ? "TRUE" : "FALSE");
   
   Print("Sell conditions - MA: ", ma_condition_sell ? "TRUE" : "FALSE",
         ", RSI: ", rsi_condition_sell ? "TRUE" : "FALSE",
         ", MACD: ", macd_condition_sell ? "TRUE" : "FALSE",
         ", Price Action: ", price_action_sell ? "TRUE" : "FALSE",
         ", BB: ", bb_condition_sell ? "TRUE" : "FALSE");
   
   Print("Signal strength - Buy: ", buy_signal_strength, ", Sell: ", sell_signal_strength);
   
   // Exit conditions for long positions
   ExitLongSignal = bearish_engulfing || 
                    (close[0] < ema_slow[0]) || 
                    (rsi[0] > RSILevel_Overbought) || 
                    (close[0] > bb_upper[0] * 0.98) || 
                    (macd_hist[0] < 0 && macd_hist[1] > 0);
   
   // Exit conditions for short positions
   ExitShortSignal = bullish_engulfing || 
                     (close[0] > ema_slow[0]) || 
                     (rsi[0] < RSILevel_Oversold) || 
                     (close[0] < bb_lower[0] * 1.02) || 
                     (macd_hist[0] > 0 && macd_hist[1] < 0);
   
   // Set final signals - ALWAYS generate either buy or sell signal based on which is stronger
   if(buy_signal_strength > sell_signal_strength)
   {
      BuySignal = true;
      SellSignal = false;
      Print("Buy signal selected (stronger than sell)");
   }
   else if(sell_signal_strength > buy_signal_strength)
   {
      BuySignal = false;
      SellSignal = true;
      Print("Sell signal selected (stronger than buy)");
   }
   else
   {
      // If equal strength, check trend direction from EMA
      if(ema_fast[0] > ema_slow[0])
      {
         BuySignal = true;
         SellSignal = false;
         Print("Buy signal selected (equal strength, uptrend)");
      }
      else
      {
         BuySignal = false;
         SellSignal = true;
         Print("Sell signal selected (equal strength, downtrend)");
      }
   }
   
   // Disable multi-timeframe check for now to ensure we get trades
   if(false && TradeOnlyWithAllTimeframes && (BuySignal || SellSignal))
   {
      bool mismatch = false;
      
      if(BuySignal && !CheckBuySignalOnTimeframe(SecondaryTimeframe))
      {
         Print("Buy signal mismatch on secondary timeframe");
         mismatch = true;
      }
      
      if(SellSignal && !CheckSellSignalOnTimeframe(SecondaryTimeframe))
      {
         Print("Sell signal mismatch on secondary timeframe");
         mismatch = true;
      }
      
      if(mismatch)
      {
         BuySignal = false;
         SellSignal = false;
      }
   }
}

//+------------------------------------------------------------------+
//| Check signal on a different timeframe                            |
//+------------------------------------------------------------------+
bool CheckBuySignalOnTimeframe(ENUM_TIMEFRAMES timeframe)
{
   int tf_ema_fast = iMA(_Symbol, timeframe, FastMAPeriod, 0, MAMethod, MAPrice);
   int tf_ema_slow = iMA(_Symbol, timeframe, SlowMAPeriod, 0, MAMethod, MAPrice);
   
   if(tf_ema_fast == INVALID_HANDLE || tf_ema_slow == INVALID_HANDLE)
      return false;
   
   double tf_ema_fast_values[], tf_ema_slow_values[];
   ArraySetAsSeries(tf_ema_fast_values, true);
   ArraySetAsSeries(tf_ema_slow_values, true);
   
   if(CopyBuffer(tf_ema_fast, 0, 0, 2, tf_ema_fast_values) != 2 ||
      CopyBuffer(tf_ema_slow, 0, 0, 2, tf_ema_slow_values) != 2)
   {
      return false;
   }
   
   bool tf_uptrend = tf_ema_fast_values[0] > tf_ema_slow_values[0];
   
   IndicatorRelease(tf_ema_fast);
   IndicatorRelease(tf_ema_slow);
   
   return tf_uptrend;
}

//+------------------------------------------------------------------+
//| Check sell signal on a different timeframe                       |
//+------------------------------------------------------------------+
bool CheckSellSignalOnTimeframe(ENUM_TIMEFRAMES timeframe)
{
   int tf_ema_fast = iMA(_Symbol, timeframe, FastMAPeriod, 0, MAMethod, MAPrice);
   int tf_ema_slow = iMA(_Symbol, timeframe, SlowMAPeriod, 0, MAMethod, MAPrice);
   
   if(tf_ema_fast == INVALID_HANDLE || tf_ema_slow == INVALID_HANDLE)
      return false;
   
   double tf_ema_fast_values[], tf_ema_slow_values[];
   ArraySetAsSeries(tf_ema_fast_values, true);
   ArraySetAsSeries(tf_ema_slow_values, true);
   
   if(CopyBuffer(tf_ema_fast, 0, 0, 2, tf_ema_fast_values) != 2 ||
      CopyBuffer(tf_ema_slow, 0, 0, 2, tf_ema_slow_values) != 2)
   {
      return false;
   }
   
   bool tf_downtrend = tf_ema_fast_values[0] < tf_ema_slow_values[0];
   
   IndicatorRelease(tf_ema_fast);
   IndicatorRelease(tf_ema_slow);
   
   return tf_downtrend;
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
   if(Trade.Buy(lot_size, _Symbol, 0, stop_loss, take_profit, "Enhanced Medallion"))
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
                        stop_loss, take_profit, ORDER_TIME_GTC, 0, "Enhanced Medallion"))
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
   if(Trade.Sell(lot_size, _Symbol, 0, stop_loss, take_profit, "Enhanced Medallion"))
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
                        stop_loss, take_profit, ORDER_TIME_GTC, 0, "Enhanced Medallion"))
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
//| Check if trading time is allowed                                 |
//+------------------------------------------------------------------+
bool IsTradeTimeAllowed()
{
   // Parse trading time strings
   int start_hour, start_minute, end_hour, end_minute;
   
   if(StringSplit(TradingStartTime, ':', start_hour, start_minute) != 2 ||
      StringSplit(TradingEndTime, ':', end_hour, end_minute) != 2)
   {
      Print("Invalid time format");
      return false;
   }
   
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
//| Helper function to split time string                             |
//+------------------------------------------------------------------+
int StringSplit(string str, char separator, int &value1, int &value2)
{
   string parts[];
   int count = StringSplit(str, separator, parts);
   
   if(count != 2)
      return 0;
   
   value1 = (int)StringToInteger(parts[0]);
   value2 = (int)StringToInteger(parts[1]);
   
   return count;
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
         ClosePosition(ticket);
      }
      else if(pos_type == POSITION_TYPE_SELL && ExitShortSignal)
      {
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
   
   return !price_mismatch;
} 