import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import kruskal
from scipy.stats import chi2_contingency
dat = pd.read_csv('./team6project/Sleep_Efficiency.csv')
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

# 시간대 30분 단위로 자르기
df['Bedtime'] = pd.to_datetime(df['Bedtime'], errors='coerce')
df['bed_halfhour'] = df['Bedtime'].dt.hour + (df['Bedtime'].dt.minute >= 30) * 0.5
df['bed_halfhour'].value_counts().sort_index()

########################################################################################
## 서론 ## ## 서론 ## ## 서론 ## ## 서론 ## ## 서론 ## ## 서론 ## ## 서론 ## ## 서론 ## ## 서론 ## 
########################################################################################

# 성별 나이대별 수면효율
## 1. 이상치 확인하기
# 연령대를 범주형으로 나누기
df['AgeGroup'] = pd.cut(df['Age'], 
                        bins=[0, 29, 39, 49, 59, 69, 100],
                        labels=['<30', '30s', '40s', '50s', '60s', '70+'])

# 박스플롯 시각화
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='AgeGroup', y='Sleep efficiency', hue='Gender')
plt.title("Sleep Efficiency by Age Group and Gender")
plt.xlabel("Age Group")
plt.ylabel("Sleep Efficiency")
plt.legend(title='Gender')
plt.tight_layout()
plt.show()


# 잠든 시간대 별 잠든 사람의 수
# 3. Bedtime 처리
df['Bedtime'] = pd.to_datetime(df['Bedtime'], errors='coerce')
df['bed_halfhour'] = df['Bedtime'].dt.hour + (df['Bedtime'].dt.minute >= 30) * 0.5

# 4. 관심 시간대 필터링 (21시 ~ 2시 30분)
valid_hours = [21, 21.5, 22, 22.5, 23, 0, 0.5, 1, 1.5, 2, 2.5]
df_filtered = df[df['bed_halfhour'].isin(valid_hours)]

# 5. bed_halfhour별 빈도수 계산
bedtime_counts = df_filtered['bed_halfhour'].value_counts().reindex(valid_hours, fill_value=0)

# 6. 시각화
plt.figure(figsize=(10, 5))
bedtime_counts.plot(kind='bar', color='salmon', edgecolor='black')
plt.title("Number of Sleep Sessions by Bedtime (30-min Intervals)")
plt.xlabel("Bedtime (Hour + 0.5 = 30min)")
plt.ylabel("Number of Sleep Records")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

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

#################################################################################
## 본론 ## 본론 ## 본론 ## 본론 ## 본론 ## 본론 ## 본론 ## 본론 ## 본론 ## 본론 ## 본론 ## 본론 
##################################################################################

# 독립변수들과 종속변수 설정
# Light sleep을 선택한 이유(다중공산성): Deep sleep을 뺀이유는 개별 상관에서는 유의했지만
# 회귀에서는 다른 변수들과 겹치는 영향(공산성) 때문에 유의하지 않게 나올수 있음
X = df[[
    'Caffeine consumption',
    'Alcohol consumption',
    'Exercise frequency',
    'Awakenings',
    'REM sleep percentage',
    'Deep sleep percentage',
    'Light sleep percentage'
]]
y = df['Sleep efficiency']
# 상수항 추가
X_const = sm.add_constant(X)
# OLS 회귀 모델 적합
model = sm.OLS(y, X_const).fit()
model.summary()

# # 운동 빈도가 수면 효율에 주는 영향
# 운동 빈도수
# 운동 빈도를 구간화해서 (예: 없음, 중간, 자주) 그룹별 
Exer_frq = df['Exercise frequency'].unique()
Exer_frq.sort()
print('운동빈도 수 : ', Exer_frq)
Ef_Lsp = df.groupby('Exercise frequency')['Light sleep percentage'].mean()
Ef_SE = df.groupby('Exercise frequency')['Sleep efficiency'].mean()

# 운동빈도에 따른 light sleep 그래프
norm_values = (Ef_Lsp .values - min(Ef_Lsp .values)) / (max(Ef_Lsp .values) - min(Ef_Lsp .values))
colors = [(0.56, 0.93, 0.56, alpha) for alpha in norm_values]
plt.bar(Ef_Lsp .index, Ef_Lsp .values, color=colors, edgecolor='black')
plt.title('Light sleep by E.F')
plt.xlabel('Frequency')
plt.ylabel('Light sleep')
plt.ylim(10,35)
plt.show()
# 운동빈도에 따른 수면 효율 그래프
norm_values = (Ef_SE.values - min(Ef_SE.values)) / (max(Ef_SE.values) - min(Ef_SE.values))
colors = [(0.56, 0.93, 0.56, alpha) for alpha in norm_values]
plt.bar(Ef_SE.index, Ef_SE.values, color=colors, edgecolor='black')
plt.title('Sleep Efficiency by E.F')
plt.xlabel('Frequency')
plt.ylabel('Sleep Efficiency')
plt.ylim(0.5,1)
plt.show()

Ef_0 = df[df['Exercise frequency']== 0]
Ef_1 = df[df['Exercise frequency']== 1]
Ef_2 = df[df['Exercise frequency']== 2]
Ef_3 = df[df['Exercise frequency']== 3]
Ef_4 = df[df['Exercise frequency']== 4]
Ef_5 = df[df['Exercise frequency']== 5]

## light sleep과 운동빈도 정규성 검정 그래프
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i in range(6):
    sns.kdeplot(
        df[df['Exercise frequency'] == i]['Light sleep percentage'].dropna(),
        ax=axes[i],
        fill=True,
        color=sns.color_palette("tab10")[i],
    )
    axes[i].set_title(f"Exercise {i}")
    axes[i].set_xlim(0, 100)
    axes[i].grid(True, linestyle='--', alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 운동빈도별 Light sleep의 정규성 검정
Ef_0_l = shapiro(Ef_0['Light sleep percentage'])
Ef_1_l = shapiro(Ef_1['Light sleep percentage'])
Ef_2_l = shapiro(Ef_2['Light sleep percentage'])
Ef_3_l = shapiro(Ef_3['Light sleep percentage'])
Ef_4_l = shapiro(Ef_4['Light sleep percentage'])
Ef_5_l = shapiro(Ef_5['Light sleep percentage'])

print('운동 0번과 light sleep 정규성 검정', Ef_0_l.pvalue )
print('운동 1번과 light sleep 정규성 검정', Ef_1_l.pvalue )
print('운동 2번과 light sleep 정규성 검정', Ef_2_l.pvalue )
print('운동 3번과 light sleep 정규성 검정', Ef_3_l.pvalue )
print('운동 4번과 light sleep 정규성 검정', Ef_4_l.pvalue )
print('운동 5번과 light sleep 정규성 검정', Ef_5_l.pvalue )
print("모든값이 매우 작은값이다, 즉 귀무가설을 기각하고 정규분포를 따르지 않는다.")

# 운동빈도별 Light sleep과 유의성 검정
result_k1 = kruskal(Ef_0['Light sleep percentage'], Ef_1['Light sleep percentage']
                 ,Ef_2['Light sleep percentage'],Ef_3['Light sleep percentage']
                 ,Ef_4['Light sleep percentage'],Ef_5['Light sleep percentage']) 
print('운동횟수에 따른 Light sleep 유의성 검정 : ', result_k1.pvalue.round(5))
print('pvalue 값이 0.0003로 작기때문에 그룹간 유의한 차이가 있다.')


## Sleep efficiency와 운동빈도 정규성 검정 그래프
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i in range(6):
    sns.kdeplot(
        df[df['Exercise frequency'] == i]['Light sleep percentage'].dropna(),
        ax=axes[i],
        fill=True,
        color=sns.color_palette("tab10")[i],
    )
    axes[i].set_title(f"Exercise {i}")
    axes[i].set_xlim(0, 70)
    axes[i].grid(True, linestyle='--', alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 운동빈도별 Sleep efficiency의 정규성 검정
Ef_0_e = shapiro(Ef_0['Sleep efficiency'])
Ef_1_e = shapiro(Ef_1['Sleep efficiency'])
Ef_2_e = shapiro(Ef_2['Sleep efficiency'])
Ef_3_e = shapiro(Ef_3['Sleep efficiency'])
Ef_4_e = shapiro(Ef_4['Sleep efficiency'])
Ef_5_e = shapiro(Ef_5['Sleep efficiency'])

print('운동 0번과 수면 효율 정규성 검정', Ef_0_e.pvalue )
print('운동 1번과 수면 효율 정규성 검정', Ef_1_e.pvalue )
print('운동 2번과 수면 효율 정규성 검정', Ef_2_e.pvalue )
print('운동 3번과 수면 효율 정규성 검정', Ef_3_e.pvalue )
print('운동 4번과 수면 효율 정규성 검정', Ef_4_e.pvalue )
print('운동 5번과 수면 효율 정규성 검정', Ef_5_e.pvalue )
print("모든값이 매우 작은값이다, 즉 귀무가설을 기각하고 정규분포를 따르지 않는다.")

# 운동빈도별 Sleep efficiency과 유의성 검정
result_k2 = kruskal(Ef_0['Sleep efficiency'], Ef_1['Sleep efficiency']
                 ,Ef_2['Sleep efficiency'],Ef_3['Sleep efficiency']
                 ,Ef_4['Sleep efficiency'],Ef_5['Sleep efficiency']) 
print('운동횟수에 따른 Light sleep 유의성 검정 : ', result_k2.pvalue)
print('pvalue 값이 매우 작기 때문에 그룹간 유의한 차이가 있다.')

##################################################################################

smokers = df[df['Smoking status']=='Yes']
nonsmokers = df[df['Smoking status']=='No']

smokers["Sleep efficiency"].mean()
nonsmokers["Sleep efficiency"].mean()


# 흡연여부 및 Light sleep간 정규성 검정
smokers["Light sleep percentage"].mean()
nonsmokers["Light sleep percentage"].mean()
sns.kdeplot(smokers['Light sleep percentage'], label='Smoker', shade=True)
sns.kdeplot(nonsmokers['Light sleep percentage'], label='Non-smoker', shade=True)
plt.title("Smoker and Non_Smoker")
plt.legend()
plt.show()
# 그림으로 그려보면 정규분포를 따르지 않는것을 알 수 있다.

# 흡연 여부별 Light sleep의 정규성 검정
from scipy.stats import shapiro
result1 = shapiro(smokers['Light sleep percentage'])
result2 = shapiro(nonsmokers['Light sleep percentage'])
print('smoker pvalue : ', result1.pvalue)
print('non_smoker pvalue : ', result2.pvalue)
print('샤피로 윌크 검정 결과 귀무가설 기각 즉, 정규성을 따르지 않는다')

# 흡연 여부별 Light sleep의 유의성 검정
from scipy.stats import mannwhitneyu
result = mannwhitneyu(smokers['Light sleep percentage'], nonsmokers['Light sleep percentage'], alternative='two-sided')
print('Light_pvalue : ', result.pvalue.round(5))
print('만휘트니 검정결과 p_value 0.00138. 따라서 흡연자와 비흡연자간의 수면 효율 데이터에 유의한 차이가 존재한다.')


# 흡연여부 및 Sleep efficiency간 정규성 검정
sns.kdeplot(smokers['Sleep efficiency'], label='Smoker', shade=True)
sns.kdeplot(nonsmokers['Sleep efficiency'], label='Non-smoker', shade=True)
plt.title("Distribution of Sleep Efficiency by Smoking Status")
plt.xlabel("Sleep Efficiency")
plt.ylabel("Density")
plt.legend()
plt.show()
# 그림으로 그려보면 정규분포를 따르지 않는것을 알 수 있다.

# 흡연 여부별 Sleep efficiency의 정규성 검정
from scipy.stats import shapiro
result1 = shapiro(smokers['Sleep efficiency'])
result2 = shapiro(nonsmokers['Sleep efficiency'])
print('pvalue : ', result1.pvalue)
print('pvalue : ', result2.pvalue)
print('샤피로 윌크 검정 결과 귀무가설 기각 즉, 정규성을 따르지 않는다')

# 흡연 여부별 Sleep efficiency의 유의성 검정
from scipy.stats import mannwhitneyu
result = mannwhitneyu(smokers['Sleep efficiency'], nonsmokers['Sleep efficiency'], alternative='two-sided')
print("p-value:", result.pvalue)
print('만휘트니 검정결과 p_value 매우작다. 따라서 흡연자와 비흡연자간의 수면 효율 데이터에 유의한 차이가 존재한다.')

##############################################################

# 1.알콜 기준 sleep efficiency 코드

# 2. Alcohol Group 컬럼 생성 (0.0 vs Other)
def alcohol_binary_group(val):
    if val == 0.0:
        return '0.0'
    else:
        return 'Other'
df['Alcohol consumption'].unique()
df['Alcohol Binary Group'] = df['Alcohol consumption'].apply(alcohol_binary_group)
# 취한 사람과 안취한 사람간의 데이터 구분

non_drink = df[df['Alcohol Binary Group'] == '0.0' ]
drink =  df[df['Alcohol Binary Group'] == 'Other' ]

# 알콜 기준 light sleep percentage

# 2. Alcohol 그룹 나누기: 0.0 vs Other
def alcohol_binary_group(val):
    if val == 0.0:
        return '0.0'
    else:
        return 'Other'

df['Alcohol Binary Group'] = df['Alcohol consumption'].apply(alcohol_binary_group)

# 3. 그룹별 평균 Light sleep percentage 계산
light_sleep_means = df.groupby('Alcohol Binary Group')['Light sleep percentage'].mean()

# 4. 막대 그래프 그리기
plt.figure(figsize=(6, 5))
light_sleep_means.loc[['0.0', 'Other']].plot(
    kind='bar',
    color=['mediumaquamarine', 'indianred'],
    edgecolor='black'
)
plt.title("Average Light Sleep Percentage by Alcohol Group (0.0 vs Other)")
plt.xlabel("Alcohol Consumption Group")
plt.ylabel("Light Sleep Percentage")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 알콜 기준 light sleep percentage 정규성 검증 
sns.kdeplot(drink['Light sleep percentage'], label='drink', shade=True)
sns.kdeplot(non_drink['Light sleep percentage'], label='non_drink', shade=True)
plt.title("Drink and Non_Drink")
plt.legend()
plt.show()

# 알콜 여부 기준 Sleep_Efficiency 샤피로 검정
result1 = shapiro(drink['Light sleep percentage'])
result2 = shapiro(non_drink['Light sleep percentage'])
print('pvalue : ', result1.pvalue)
print('pvalue : ', result2.pvalue)
print('샤피로 윌크 검정 결과 귀무가설 기각 즉, 정규성을 따르지 않는다')

# 알콜 여부 기준 Sleep_Efficiency 유의성 검정
result = mannwhitneyu(drink['Light sleep percentage'], non_drink['Light sleep percentage'], alternative='two-sided')
print("p-value:", result.pvalue)
print('만휘트니 검정결과 p_value 매우작다. 따라서 흡연자와 비흡연자간의 수면 효율 데이터에 유의한 차이가 존재한다.')


# 3. 그룹별 Sleep efficiency 평균 계산
group_means = df.groupby('Alcohol Binary Group')['Sleep efficiency'].mean()

# 4. 막대 그래프 시각화
plt.figure(figsize=(6, 5))
group_means.loc[['0.0', 'Other']].plot(
    kind='bar',
    color=['royalblue', 'lightcoral'],
    edgecolor='black'
)
plt.title("Average Sleep Efficiency by Alcohol Group (0.0 vs Other)")
plt.xlabel("Alcohol Consumption Group")
plt.ylabel("Sleep Efficiency")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 알콜 여부 기준 Sleep_Efficiency 정규성 검증 
sns.kdeplot(drink['Sleep efficiency'], label='drink', shade=True)
sns.kdeplot(non_drink['Sleep efficiency'], label='non_drink', shade=True)
plt.title("Drink and Non_Drink")
plt.legend()
plt.show()

# 알콜 여부 기준 Sleep_Efficiency 샤피로 검정
result1 = shapiro(drink['Sleep efficiency'])
result2 = shapiro(non_drink['Sleep efficiency'])
print('pvalue : ', result1.pvalue)
print('pvalue : ', result2.pvalue)
print('샤피로 윌크 검정 결과 귀무가설 기각 즉, 정규성을 따르지 않는다')

# 알콜 여부 기준 Sleep_Efficiency 유의성 검정
result = mannwhitneyu(drink['Sleep efficiency'], non_drink['Sleep efficiency'], alternative='two-sided')
print("p-value:", result.pvalue)
print('만휘트니 검정결과 p_value 매우작다. 따라서 흡연자와 비흡연자간의 수면 효율 데이터에 유의한 차이가 존재한다.')

## 카페인 변수 넣었던 이유
# 파생 변수 생성 (카이제곱용 이진화)
df['High_Caffeine'] = (df['Caffeine consumption'] > 3).astype(int)      # 3단위 초과 → 고카페인
df['Short_Sleep'] = (df['Sleep duration'] < 6).astype(int)              # 6시간 미만 → 짧은 수면
df['High_Alcohol'] = (df['Alcohol consumption'] > 3).astype(int)        # 3단위 초과 → 고알콜
df['Low_Efficiency'] = (df['Sleep efficiency'] < 0.85).astype(int)      # 85% 미만 → 수면 효율 낮음
df['High_Exercise'] = (df["Exercise frequency"] > 2 ).astype(int)       # 3번 이상 -> 많은 운동량
df['Low_lisht_sleep'] = (df['Light sleep percentage'] < 25).astype(int)

# 카이제곱 검정할 변수쌍 정의
pairs = [
    ('High_Caffeine', 'Short_Sleep'),
    ('High_Caffeine', 'High_Alcohol'),
    ('High_Caffeine', 'Low_Efficiency'),
    ('High_Caffeine', 'High_Exercise'),
    ('High_Caffeine', 'Low_lisht_sleep')
]

# 4. 교차표 + p-value 출력
for var1, var2 in pairs:
    print(f"\n🔹 교차표: {var1} vs {var2}")
    table = pd.crosstab(df[var1], df[var2])
    print(table)
    stat, p, dof, expected = chi2_contingency(table)
    print(f"➡ p-value: {p:.5f}")

# 카이제곱 검정 실행
results = []

for var1, var2 in pairs:
    table = pd.crosstab(df[var1], df[var2])
    stat, p, dof, expected = chi2_contingency(table)
    results.append({
        "Variable 1": var1,
        "Variable 2": var2,
        "p-value": round(p, 5)
    })

# 결과 보기
results_df = pd.DataFrame(results)
print(results_df)

#################################################################################
## 본론2 ## 본론2 ## 본론2 ## 본론2 ## 본론2 ## 본론2 ## 본론2 ## 본론2 ## 본론2 ## 본론2 ## 본론2 ## 본론2 
##################################################################################

# 원래 수치형 변수들
num_vars = ['Light sleep percentage', 'Alcohol consumption','Awakenings', 'Exercise frequency', 'Caffeine consumption']

# 범주형 변수 → 더미로 변환
smoking_dummy = pd.get_dummies(df['Smoking status'], drop_first=True)

# 최종 X 구성 (수치형 + 더미 합치기)
X = pd.concat([df[num_vars], smoking_dummy], axis=1)
X = X.astype(float)
X = X.dropna()
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
# 변수로 다양할때 등분산성 검정은 Breusch-Pegan 사용
bp_test = het_breuschpagan(model.resid, X_const)
bp_p = bp_test[1]
print(f"\nBreusch-Pagan Test (잔차 등분산성 검정):\np-value = {bp_p:.5f}")


# Q-Q Plot 그리기
plt.figure(figsize=(5, 5))
sm.qqplot(model.resid, line='45', fit=True)
_=plt.title("Q-Q Plot of Residuals (Based on Scatter Pattern)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 잔차분산성 검증
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(0, color='red')
plt.title("Fitted vs Residuals")





### 최적 모델 만드는 과정
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# 원래 수치형 변수들
num_vars = ['Light sleep percentage', 'Alcohol consumption','Awakenings', 'Exercise frequency', 'Caffeine consumption']

# 범주형 변수 → 더미로 변환
smoking_dummy = pd.get_dummies(df['Smoking status'], drop_first=True)

# 최종 X 구성 (수치형 + 더미 합치기)
X = pd.concat([df[num_vars], smoking_dummy], axis=1)
X = X.astype(float)
X = X.dropna()
y = df['Sleep efficiency']
names = X.columns.tolist()

# AIC 계산 함수 (오류 없는 버전)
def aic_score(estimator, X_subset_np, y_np): 
    X_const = sm.add_constant(X_subset_np)
    model = sm.OLS(y_np, X_const).fit()
    return -model.aic  # AIC가 낮을수록 좋으므로 음수 반환

# 모델 정의
lr = LinearRegression()

# SFS (AIC 기준 forward selection)
sfs = SFS(estimator=lr,
          k_features=(1, len(X.columns)),
          forward=True,
          floating=False,
          scoring=aic_score,
          cv=0,
          verbose=1)

# 피팅
sfs.fit(X.values, y.values)

# 선택된 변수 추출
selected_indices = list(sfs.k_feature_idx_)
selected_features = [names[i] for i in selected_indices]
print("최종 선택된 변수:", selected_features)

# 최종 모델 학습 (statsmodels OLS)
X_selected = sm.add_constant(X[selected_features])
final_model = sm.OLS(y, X_selected).fit()

# 결과 출력
print(final_model.summary())# 원래 수치형 변수들


### 최종모델 검증하는 방법
# 1. 정규성 (Shapiro-Wilk)
shapiro_stat, shapiro_p = shapiro(final_model.resid)

# 2. 등분산성 (Breusch-Pagan)
bp_test = het_breuschpagan(final_model.resid, X_selected)
bp_p = bp_test[1]

# 3. Q-Q Plot
sm.qqplot(final_model.resid, line='45', fit=True)
plt.title("Q-Q Plot of Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()

### (해석)
# 잔차는 대체로 정규성을 만족하지만, 극단값 구간에서는 일부 왜곡이 나타난다.
# 이는 회귀계수의 정확도에 큰 영향을 주지는 않으나,
# 정밀한 신뢰구간 해석이나 극단값에 민감한 분석에서는 주의가 필요하다



# 4. 테스트 데이터 없으므로 rsquared_adj 확인인
print(final_model.rsquared_adj)

# 4. 결과 출력
print("선택된 변수:", selected_features)
print(f"Shapiro-Wilk p-value (정규성): {shapiro_p:.5f}")
print(f"Breusch-Pagan p-value (등분산성): {bp_p:.5f}")
print("회귀 요약:")
print(final_model.summary())


### (전체 해석)
# 잔차가 정규분포를 따르지 않음
# 하지만 회귀계수 자체가 나쁜 건 아님 예측력 자체에는 큰 영향 없음
# 등분산성에서는 잔차의 분산성이 일정하지 않음
# 테스트 데이터가 없으므로 rsquared_adj를 확인한 결과 나쁘지 않은 설명력을 가짐

### (결론)
# 이 모델은 이분산성이 존재해서 통계적 해석(p값, t검정)은 신뢰하기 어렵지만,
# 왜냐하면 수면 효율성, rem, deep, light sleep은 매우 서로 연관성이 높기 때문이다.
# 우리 데이터로는 회귀예측을 할 수 없다. 군집분석, 주성분 분석, kluster으로 모델예측이 가능하다
# 예측 성능이 괜찮다면 예측용으로는 여전히 활용 가능함.
# 다른 회귀 모델을 반영하거나 혹은 데이터 변환이 필요해보임

from statsmodels.stats.anova import anova_lm
anova_lm(final_model,model)
###(해석)
# 추가된 변수는 기존 모델 대비 수면 효율 예측에 통계적으로 유의한 기여를 하지 않았으며
# 두 모델 간 차이는 F(1, 446) = 0.47, p = 0.493로 나타나,
# final_model이 충분한 설명력을 유지함을 의미함.