import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan

dat = pd.read_csv('./data/Sleep_Efficiency.csv')
df = pd.DataFrame(dat)

# 변수 정의
# ID  : 식별자
# Age : 나이
# Gender : 성별
# Bedtime : 취침시간
# Wakeup time : 기상시간
# Sleep duration : 총수면시간(단위 : 시간)
# Sleep efficiency : 수면 효율(침대에 누워있을시간 대비 잠자는 시간)
# REM sleep percentage : 램 수면 시간 비율
# Deep sleep percentage : 깊은 수면 시간 비율
# Light sleep percentage : 얕은 수면 시간 비율 
# Awakenings : 깬 횟수
# Caffeine consumption : 카페인 셥취량
# Alcohol consumption : 알코올 섭취량
# Smoking status : 흡연 여부
# Exercise frequency : 운동 빈도


# 데이터 알아보기
df.isnull().sum()

# Awakenings, Caffeine consumption,Alcohol consumption,Exercise frequency에서 nan값 발생
# 각각 nan값은 최빈값으로 대체하기 
df['Awakenings'].mode()     # 최빈값 1
df['Caffeine consumption'].mode()  # 최빈값 0
df['Alcohol consumption'].mode()   # 최빈값 0
df['Exercise frequency'].mode()   # 최빈값 3

df['Awakenings'] = df['Awakenings'].fillna(1)
df['Caffeine consumption'] = df['Caffeine consumption'].fillna(0)
df['Alcohol consumption'] = df['Alcohol consumption'].fillna(0)
df['Exercise frequency'] = df['Exercise frequency'].fillna(3)


### 히트맵 코드
# 수치형 변수만 선택
numeric_df = df.select_dtypes(include='number')
# 상관계수 행렬 계산
corr_matrix = numeric_df.corr()
# 수치형 변수만 선택
numeric_df = df.select_dtypes(include='number')
# 상관계수 행렬 계산
corr_matrix = numeric_df.corr()
# 히트맵 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, 
            annot=True,         # 셀 안에 숫자 표시
            fmt=".2f",          # 소수점 둘째자리까지
            cmap="coolwarm",    # 색상 맵
            square=True, 
            linewidths=0.5)
plt.title("수치형 변수 간 상관계수 히트맵")
plt.tight_layout()
plt.grid(False)
plt.show()



# 운동 빈도가 수면 효율에 주는 영향
# 운동 빈도수
# 운동 빈도를 구간화해서 (예: 없음, 중간, 자주) 그룹별 
Exer_frq = df['Exercise frequency'].unique()
Exer_frq.sort()
print('운동빈도 수 : ', Exer_frq)

# 데이터 그룹화 
Ef_Sd = df.groupby('Exercise frequency')['Sleep duration'].mean()
Ef_Dsp = df.groupby('Exercise frequency')['Deep sleep percentage'].mean()
# Ef_Awk = df.groupby('Exercise frequency')['Awakenings'].mean()
# Ef_SE = df.groupby('Exercise frequency')['Awakenings'].mean()
Ef_Lsp = df.groupby('Exercise frequency')['Light sleep percentage'].mean()
Ef_SE = df.groupby('Exercise frequency')['Sleep efficiency'].mean()

norm_values = (Ef_Lsp .values - min(Ef_Lsp .values)) / (max(Ef_Lsp .values) - min(Ef_Lsp .values))
colors = [(0.56, 0.93, 0.56, alpha) for alpha in norm_values]
plt.bar(Ef_Lsp .index, Ef_Lsp .values, color=colors, edgecolor='black')
plt.title('Light sleep by E.F')
plt.xlabel('Frequency')
plt.ylabel('Light sleep')
plt.ylim(10,35)
plt.show()

norm_values = (Ef_SE.values - min(Ef_SE.values)) / (max(Ef_SE.values) - min(Ef_SE.values))
colors = [(0.56, 0.93, 0.56, alpha) for alpha in norm_values]
plt.bar(Ef_SE.index, Ef_SE.values, color=colors, edgecolor='black')
plt.title('Sleep Efficiency by E.F')
plt.xlabel('Frequency')
plt.ylabel('Sleep Efficiency')
plt.ylim(0.5,1)
plt.show()

## light sleep과 운동빈도 정규성 검정
model = ols("Q('Light sleep percentage') ~ Q('Exercise frequency')", data=df).fit()
residuals = model.resid
sm.qqplot(residuals, line='45', fit=True)
plt.title("Q-Q Plot of Residuals")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True)
plt.show()

model = ols("Q('Sleep efficiency') ~ Q('Exercise frequency')", data=df).fit()
residuals = model.resid
sm.qqplot(residuals, line='45', fit=True)
plt.title("Q-Q Plot of Residuals")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True)
plt.show()




norm_values = (Ef_Sd.values - min(Ef_Sd.values)) / (max(Ef_Sd.values) - min(Ef_Sd.values))
colors = [(0.56, 0.93, 0.56, alpha) for alpha in norm_values]
plt.bar(Ef_Sd.index, Ef_Sd.values, color=colors, edgecolor='black')
plt.title('Sleep duration by E.F')
plt.xlabel('Frequency')
plt.ylabel('Sleep duration')
plt.ylim(5, 8)
plt.show()
# 평균 수면시간으로 시각화 하면 최대 시간 - 최소 시간이 0.36으로 큰 차이가 없다.
# 운동횟수는 총 수면시간과 큰 영향이 없다.

norm_values = (Ef_Dsp.values - min(Ef_Dsp.values)) / (max(Ef_Dsp.values) - min(Ef_Dsp.values))
colors = [(0.56, 0.93, 0.56, alpha) for alpha in norm_values]
plt.bar(Ef_Dsp.index, Ef_Dsp.values, color=colors, edgecolor='black')
plt.title('Deep sleep by E.F')
plt.xlabel('Frequency')
plt.ylabel('Deep sleep %')
plt.ylim(30, 60)
plt.show()
# 하지만 수면 효율는 최대 최소값 차이가 11.2%로 유의한 차이가 있다.
# 따라서 수면 효율에 중심을 두고 분석을 시작했다.

# norm_values = (max(Ef_Awk.values) - Ef_Awk.values) / (max(Ef_Awk.values) - min(Ef_Awk.values))
# colors = [(0.56, 0.93, 0.56, alpha) for alpha in norm_values]
# plt.bar(Ef_Awk.index, Ef_Awk.values, color=colors, edgecolor='black')
# plt.title('Awakenings by E.F')
# plt.xlabel('Frequency')
# plt.ylabel('Awakenings')
# plt.ylim(0, 2)
# plt.show()
# 깨어난 횟수가 흡연과는 상관이 없지만 깨어난횟수와 효율은 상관관계가 있고 또 깨어난 횟수는 
# 운동과 음의 상관관계가 나타나기 때문에 효율, 깨어난횟수, 운동의 관계가 있을것이다.



####
### light랑 Sleep efficiency의 회귀분석모형
# 독립변수와 종속변수 지정 : 카페인 포함 버전
X = df[['Light sleep percentage', 'Alcohol consumption','Awakenings', 'Exercise frequency', 'Caffeine consumption']]
y = df['Sleep efficiency']
X_const = sm.add_constant(X)

# 회귀 모델 적합
model = sm.OLS(y, X_const).fit()

# 모델 결과 출력
print(model.summary())

# 정규성 검정 (Shapiro-Wilk)
shapiro_stat, shapiro_p = shapiro(model.resid)
print(f"\nShapiro-Wilk Test (잔차 정규성 검정):\nStatistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.5f}")

# 등분산성 검정 (Breusch-Pagan)
bp_test = het_breuschpagan(model.resid, X_const)
bp_p = bp_test[1]   
print(f"\nBreusch-Pagan Test (잔차 등분산성 검정):\np-value = {bp_p:.5f}")

# 다중공선성 확인 (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_df = pd.DataFrame()
vif_df["feature"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVIF (Variance Inflation Factor):")
print(vif_df)


# 독립변수와 종속변수 지정 : 카페인 포함 미포함버전
X = df[['Light sleep percentage', 'Alcohol consumption','Awakenings', 'Exercise frequency']]
y = df['Sleep efficiency']
X_const = sm.add_constant(X)

# 회귀 모델 적합
model2 = sm.OLS(y, X_const).fit()

# 모델 결과 출력
print(model2.summary())

# 정규성 검정 (Shapiro-Wilk)
shapiro_stat, shapiro_p = shapiro(model.resid)
print(f"\nShapiro-Wilk Test (잔차 정규성 검정):\nStatistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.5f}")

# 등분산성 검정 (Breusch-Pagan)
bp_test = het_breuschpagan(model.resid, X_const)
bp_p = bp_test[1]
print(f"\nBreusch-Pagan Test (잔차 등분산성 검정):\np-value = {bp_p:.5f}")


from statsmodels.stats.anova import anova_lm
anova_lm(model, model2)
# Caffeine 변수는 회귀모형에 추가했을 때 유의하지 않고, 오히려 성능이 더 나빠졌다.  


