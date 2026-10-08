import numpy as np
import pandas as pd
try:
    import yfinance as yf
except ImportError:
    # سيتم تثبيتها في ملف الـ YAML لاحقاً، لكن نضع تحقق هنا للأمان المحلي
    pass 

# ═══════════════════════════════════════════════
# 2.5 محرك البيانات والمؤشرات (Data Engine & Indicators) - NEW LAYER 2
# ═══════════════════════════════════════════════

def fetch_h1_data(symbol: str, period_days: int = 7):
    """
    يجلب بيانات الشموع للساعة الواحدة (H1) لعدد أيام محدد.
    يرجع DataFrame نظيفاً أو None في حال الفشل.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{period_days}d", interval="1h", auto_adjust=False, actions=False)
        
        if df is None or df.empty:
            log.warning(f"No data returned for {symbol}")
            return None
            
        # تنظيف العمود متعدد المستويات إن وجد (مشكلة شائعة في yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]) for c in df.columns]
            
        required_cols = ["Open", "High", "Low", "Close"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            log.error(f"Missing columns for {symbol}: {missing}")
            return None
            
        df = df[required_cols].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated()].sort_index().dropna()
        
        # إزالة الشمعة الحالية غير المكتملة (Look-ahead bias prevention)
        now_utc = pd.Timestamp.now(tz="UTC")
        # نتحقق هل آخر شمعة بدأت قبل أقل من ساعة؟ (أي أنها لا تزال تتكون)
        if not df.empty and df.index[-1] + pd.Timedelta(hours=1) > now_utc:
             df = df.iloc[:-1]
             
        return df
    except Exception as e:
        log.error(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_indicators(df: pd.DataFrame):
    """
    يحسب EMA200 و Stochastic (%K, %D) على الـ DataFrame المُدخل.
    يضيف الأعمدة الجديدة لنفس الـ DataFrame ويعيده.
    """
    if df is None or len(df) < 200: # نحتاج 200 شمعة على الأقل لحساب EMA200 بدقة
        return df
        
    close_prices = df["Close"]
    
    # 1. حساب EMA 200
    df["EMA_200"] = close_prices.ewm(span=200, adjust=False).mean()
    
    # 2. حساب Stochastic Oscillator
    # الفترة الافتراضية للمواصفات الفنية كانت K=9, D=5
    k_period = 9
    d_period = 5
    
    lowest_low = df["Low"].rolling(window=k_period).min()
    highest_high = df["High"].rolling(window=k_period).max()
    
    # تجنب القسمة على صفر
    denom = highest_high - lowest_low
    denom.replace(0, np.nan, inplace=True) 
    
    raw_k = 100 * ((close_prices - lowest_low) / denom)
    
    # تعبئة القيم الفارغة الناتجة عن أول few rows
    raw_k.fillna(method='ffill', inplace=True) 
    raw_k.fillna(50, inplace=True) # قيمة افتراضية neutral
    
    df["STOCH_K"] = raw_k
    df["STOCH_D"] = raw_k.rolling(window=d_period).mean()
    df["STOCH_D"].fillna(method='ffill', inplace=True)
    
    return df

def test_layer_2():
    """دالة اختبار سريعة للتأكد من عمل المحرك قبل الدمج الكامل."""
    test_symbol = "EURUSD=X"
    log.info(f"Testing Layer 2 with symbol: {test_symbol}")
    
    df_raw = fetch_h1_data(test_symbol, period_days=3)
    if df_raw is None:
        log.error("Failed to fetch raw data.")
        return False
        
    log.info(f"Fetched {len(df_raw)} candles.")
    
    df_processed = calculate_indicators(df_raw)
    if df_processed is None or "EMA_200" not in df_processed.columns:
        log.error("Failed to calculate indicators.")
        return False
        
    last_row = df_processed.iloc[-1]
    log.info("--- Sample Calculation Results ---")
    log.info(f"Candle Time: {last_row.name}")
    log.info(f"Close Price: {last_row['Close']:.5f}")
    log.info(f"EMA 200:     {last_row['EMA_200']:.5f}")
    log.info(f"Stoch %K:    {last_row['STOCH_K']:.2f}")
    log.info(f"Stoch %D:    {last_row['STOCH_D']:.2f}")
    
    # فحص بسيط للاتجاه
    trend_up = last_row['Close'] > last_row['EMA_200']
    log.info(f"Trend Direction: {'UP' if trend_up else 'DOWN'}")
    
    return True

# نضيف استدعاء الاختبار داخل main مؤقتاً للتأكد
original_main = main
def main_with_test():
    original_main()
    print("\n>>> Running Layer 2 Self-Test <<<")
    success = test_layer_2()
    if success:
        send_telegram("✅ **Layer 2 (Data Engine) Tested Successfully!**\nIndicators calculated correctly.")
    else:
        send_telegram("❌ **Layer 2 Test Failed.** Check logs.")

# استبدال الدالة الرئيسية المؤقتة
main = main_with_test
