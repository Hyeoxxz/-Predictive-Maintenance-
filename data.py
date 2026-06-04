import numpy as np
import pandas as pd

np.random.seed(42)

def generate_data(n_engines=150):
    records = []
    
    for engine_id in range(1, n_engines + 1):
        # 엔진마다 수명이 다름 (150~300 사이클)
        total_life = np.random.randint(150, 300)
        
        for cycle in range(1, total_life + 1):
            # 열화도: 사이클이 진행될수록 0 → 1 에 가까워짐
            degradation = cycle / total_life
            
            # 센서 11개 (시간이 갈수록 값이 변함)
            s1  = 518.67 + np.random.normal(0, 0.5)
            s2  = 642.68 + degradation * 8  + np.random.normal(0, 0.5)
            s3  = 1590.0 + degradation * 20 + np.random.normal(0, 2.0)
            s4  = 1408.0 + degradation * 15 + np.random.normal(0, 2.0)
            s5  = 14.62  + np.random.normal(0, 0.01)
            s6  = 21.61  + np.random.normal(0, 0.01)
            s7  = 554.36 - degradation * 5  + np.random.normal(0, 0.3)
            s8  = 2388.0 - degradation * 10 + np.random.normal(0, 1.0)
            s9  = 9046.0 - degradation * 30 + np.random.normal(0, 5.0)
            s10 = 1.3    + degradation * 0.05 + np.random.normal(0, 0.005)
            s11 = 47.47  + degradation * 2  + np.random.normal(0, 0.2)
            
            rul = total_life - cycle  # 잔여 수명
            
            records.append({
                'engine_id': engine_id,
                'cycle': cycle,
                'sensor_1': round(s1, 4),
                'sensor_2': round(s2, 4),
                'sensor_3': round(s3, 4),
                'sensor_4': round(s4, 4),
                'sensor_5': round(s5, 4),
                'sensor_6': round(s6, 4),
                'sensor_7': round(s7, 4),
                'sensor_8': round(s8, 4),
                'sensor_9': round(s9, 4),
                'sensor_10': round(s10, 4),
                'sensor_11': round(s11, 4),
                'RUL': rul,
                'failure_within_30': 1 if rul <= 30 else 0
            })
    
    return pd.DataFrame(records)

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("sensor_data.csv", index=False)
    print(f"완료: {len(df):,}행, {df['engine_id'].nunique()}개 엔진")
    print(f"고장 임박 비율: {df['failure_within_30'].mean():.1%}")
    print(df.head(3))