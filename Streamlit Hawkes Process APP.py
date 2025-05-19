import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
import torch
import torch.optim as optim
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 페이지 기본 설정
st.set_page_config(
    page_title="🧊 IP 이상치 탐지 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS 적용
st.markdown("""
    <style>
        .stApp {
            background-color: #FFF0F5;
        }
        .stButton>button {
            background-color: #FF69B4;
            color: white;
        }
        .stSelectbox {
            background-color: #FFE4E1;
        }
        .css-1d391kg {
            background-color: #FFB6C1;
        }
        .sidebar .sidebar-content {
            background-color: #FFC0CB;
        }
    </style>
""", unsafe_allow_html=True)


# 전처리 함수 정의
def preprocess_edr_data(df, user_col, proc_col, time_col, bytes_col, impute_method='knn'):
    df_processed = df.copy()

    # 레이블 인코딩 - 사용자 및 프로세스를 숫자로 변환
    user_encoder = LabelEncoder()
    proc_encoder = LabelEncoder()

    df_processed[user_col] = user_encoder.fit_transform(df_processed[user_col])
    df_processed[proc_col] = proc_encoder.fit_transform(df_processed[proc_col])

    user_mapping = dict(zip(user_encoder.classes_, range(len(user_encoder.classes_))))
    proc_mapping = dict(zip(proc_encoder.classes_, range(len(proc_encoder.classes_))))

    # 시간 데이터 처리
    df_processed[time_col] = pd.to_datetime(df_processed[time_col])
    df_processed[time_col] = (df_processed[time_col] - df_processed[time_col].min()).dt.total_seconds()

    # 결측치 처리는 유지 (필요한 경우)
    if impute_method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
        df_processed[bytes_col] = imputer.fit_transform(df_processed[[bytes_col]])

    # 정규화 코드 제거 - bytes_col을 원본 그대로 사용

    return df_processed, user_mapping, proc_mapping, user_encoder, proc_encoder, None  # scaler는 None으로 반환


class EDR_OnlineDMHP(torch.nn.Module):
    def __init__(self, num_users, num_processes, **params):
        super(EDR_OnlineDMHP, self).__init__()
        # 기본 파라미터 설정
        self.default_params = {
            'feature_dim': 1,
            'hidden_dim': 256,
            'lr': 0.0001,
            'time_window': 48,
            'max_history_length': 5000,
            'warm_up_period': 2000,
            'update_interval': 50,
            'checkpoint_interval': 5000,
            'threshold': 3.0,
            'decay_rate': 0.05,
            'min_events_for_anomaly': 10
        }
        # 사용자 지정 파라미터로 기본값 업데이트
        self.params = {**self.default_params, **params}

        self.num_users = num_users
        self.num_processes = num_processes
        self.feature_dim = self.params['feature_dim']
        self.hidden_dim = self.params['hidden_dim']

        # 임베딩 레이어
        self.user_embedding = torch.nn.Embedding(num_users, self.hidden_dim * 2)
        self.process_embedding = torch.nn.Embedding(num_processes, self.hidden_dim * 2)

        # BatchNorm1d를 제거하고 Layer Normalization으로 대체
        self.fc1 = torch.nn.Linear(self.hidden_dim * 4 + self.feature_dim, self.hidden_dim * 2)
        self.ln1 = torch.nn.LayerNorm(self.hidden_dim * 2)
        self.dropout1 = torch.nn.Dropout(0.3)

        self.fc2 = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ln2 = torch.nn.LayerNorm(self.hidden_dim)
        self.dropout2 = torch.nn.Dropout(0.2)

        self.fc3 = torch.nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.ln3 = torch.nn.LayerNorm(self.hidden_dim // 2)

        self.fc4 = torch.nn.Linear(self.hidden_dim // 2, 1)

    def forward(self, user_ids, proc_ids, features):
        user_emb = self.user_embedding(user_ids)
        process_emb = self.process_embedding(proc_ids)
        combined = torch.cat([user_emb, process_emb, features], dim=1)

        x = self.fc1(combined)
        x = self.ln1(x)  # BatchNorm을 LayerNorm으로 변경
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.ln2(x)  # BatchNorm을 LayerNorm으로 변경
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.ln3(x)  # BatchNorm을 LayerNorm으로 변경
        x = torch.relu(x)

        intensity = torch.exp(self.fc4(x))

        return intensity


# edr_online_train_and_detect 함수 수정
def edr_online_train_and_detect(df, num_users, num_processes, user_encoder, proc_encoder, original_df, **params):
    # 기본 파라미터 설정
    default_params = {
        'feature_dim': 1,
        'hidden_dim': 256,
        'lr': 0.0001,
        'time_window': 48,
        'max_history_length': 5000,
        'warm_up_period': 2000,
        'update_interval': 50,
        'checkpoint_interval': 5000,
        'threshold': 60.0,  # 임계값 기본값 수정 (0~100 기준)
        'decay_rate': 0.05,
        'min_events_for_anomaly': 10,
        'user_col': 'user_id',
        'proc_col': 'process_id',
        'time_col': 'timestamp',
        'bytes_col': 'bytes',
        'score_scale': 1.5,  # 시그모이드 기울기 조절 (낮을수록 더 가파른 기울기)
        'score_max': 100.0  # 원본 점수의 최대 제한값
    }

    # 사용자 지정 파라미터로 기본값 업데이트
    params = {**default_params, **params}

    # 모든 사용자(IP)를 가져오기
    unique_users = df[params['user_col']].unique()

    # IP별 모델, 최적화기, 이력 및 손실 관리를 위한 딕셔너리 생성
    models = {}
    optimizers = {}
    histories = {}
    all_losses = {}

    # 이상치 점수 정규화를 위한 이동 윈도우 사용
    recent_scores = []  # 최근 원본 이상치 점수를 저장할 리스트
    score_window_size = 1000  # 점수 정규화에 사용할 윈도우 크기

    # 각 IP별로 별도의 모델 및 관련 객체 초기화
    for user_id in unique_users:
        models[user_id] = EDR_OnlineDMHP(num_users, num_processes, **params)
        optimizers[user_id] = optim.Adam(models[user_id].parameters(), lr=params['lr'])
        histories[user_id] = []
        all_losses[user_id] = []

    # 전체 결과를 저장할 리스트
    results = []

    # 각 IP별 처리된 이벤트 카운트 (웜업 확인용)
    event_counts = {user_id: 0 for user_id in unique_users}

    # 진행 상태 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_text = st.empty()

    start_time = time.time()
    initial_time = pd.Timestamp(df[params['time_col']].iloc[0])

    for i in range(len(df)):
        if i % 100 == 0:
            current_time = time.time()
            progress = (i + 1) / len(df)
            progress_bar.progress(progress)

            elapsed_time = current_time - start_time
            time_per_item = elapsed_time / (i + 1) if i > 0 else 0
            remaining_items = len(df) - (i + 1)
            estimated_remaining = time_per_item * remaining_items

            status_text.text(
                f"처리 중... {i + 1:,}/{len(df):,} ({progress:.1%})\n"
                f"경과 시간: {elapsed_time / 60:.1f}분 | "
                f"예상 남은 시간: {estimated_remaining / 60:.1f}분"
            )

        current_data = df.iloc[i]
        original_data = original_df.iloc[i]  # 원본 데이터(인코딩 전)에 접근

        # 현재 데이터의 사용자 ID
        user_id = current_data[params['user_col']]
        user_id_tensor = torch.tensor([user_id], dtype=torch.long)
        proc_id = torch.tensor([current_data[params['proc_col']]], dtype=torch.long)
        feature = torch.tensor([[current_data[params['bytes_col']]]], dtype=torch.float)

        # 현재 사용자의 이벤트 카운트 증가
        event_counts[user_id] += 1

        event_time = pd.Timestamp(current_data[params['time_col']])
        current_time = (event_time - initial_time).total_seconds() / 3600

        # 현재 사용자의 이력 관리
        history = histories[user_id]
        history = [h for h in history if (current_time - h['time']) <= params['time_window']]

        if len(history) > 0:
            historical_influence = 0
            for h in history:
                time_diff = current_time - h['time']
                decay = np.exp(-params['decay_rate'] * time_diff)
                historical_influence += decay
        else:
            historical_influence = 0

        # 현재 사용자의 모델에 대한 예측
        model = models[user_id]
        intensity = model(user_id_tensor, proc_id, feature)

        # IP별 웜업 기간 확인 후 이상치 탐지 (웜업 기간 이후에만)
        if event_counts[user_id] > params['warm_up_period']:
            proc_history = [h for h in history if h['proc_id'].item() == proc_id.item()]

            if len(history) >= params['min_events_for_anomaly']:
                # intensity 값이 너무 작을 경우를 대비해 클리핑
                clipped_intensity = torch.clamp(intensity, min=1e-10)
                base_score = torch.abs(1.0 / clipped_intensity).item()  # 강도의 역수를 사용

                # 사용자 및 프로세스별 통계 계산
                user_features = [h['feature'].item() for h in history]
                user_mean = np.mean(user_features)
                user_std = np.std(user_features) + 1e-6

                proc_features = [h['feature'].item() for h in proc_history] if proc_history else user_features
                proc_mean = np.mean(proc_features)
                proc_std = np.std(proc_features) + 1e-6

                # 정규화된 특성값 계산
                current_feature = feature.item()
                normalized_feature = abs(current_feature - user_mean) / user_std
                proc_normalized = abs(current_feature - proc_mean) / proc_std

                # 기본 이상치 점수 계산 - 가중치 조정으로 점수 범위 확대
                raw_score = base_score * \
                            (1 + np.log1p(historical_influence)) * \
                            (1 + np.sqrt(normalized_feature)) * \
                            (1 + np.sqrt(proc_normalized))

                # 점수 클리핑 - 너무 큰 값 방지
                raw_score = min(raw_score, params['score_max'])

                # 최근 점수 윈도우에 추가
                recent_scores.append(raw_score)
                if len(recent_scores) > score_window_size:
                    recent_scores.pop(0)  # 가장 오래된 점수 제거

                # 점수 정규화 및 변환 - 더 넓은 분포 생성
                if len(recent_scores) > 100:
                    # 현재 점수의 백분위 계산 (0~1)
                    percentile = sum(1 for s in recent_scores if s <= raw_score) / len(recent_scores)

                    # 백분위 기반 스케일링 (0-100)
                    if percentile > 0.99:
                        # 상위 1%는 90-100점 범위
                        anomaly_score = 90 + (percentile - 0.99) * 1000
                    elif percentile > 0.95:
                        # 상위 1-5%는 80-90점 범위
                        anomaly_score = 80 + (percentile - 0.95) * 250
                    elif percentile > 0.9:
                        # 상위 5-10%는 70-80점 범위
                        anomaly_score = 70 + (percentile - 0.9) * 200
                    elif percentile > 0.8:
                        # 상위 10-20%는 60-70점 범위
                        anomaly_score = 60 + (percentile - 0.8) * 100
                    elif percentile > 0.7:
                        # 상위 20-30%는 50-60점 범위
                        anomaly_score = 50 + (percentile - 0.7) * 100
                    elif percentile > 0.5:
                        # 상위 30-50%는 30-50점 범위
                        anomaly_score = 30 + (percentile - 0.5) * 100
                    else:
                        # 하위 50%는 0-30점 범위
                        anomaly_score = percentile * 60
                else:
                    # 충분한 데이터가 쌓이기 전에는 시그모이드 함수로 변환
                    sigmoid_score = 1.0 / (1.0 + np.exp(-raw_score / params['score_scale']))
                    anomaly_score = sigmoid_score * 100.0

                # 최종 점수 제한 (0-100)
                anomaly_score = max(0, min(anomaly_score, 100.0))

                # 이상치 판단 (정규화된 점수 기준)
                is_anomaly = anomaly_score > params['threshold']

                # 인코딩된 값 대신 원본 값 사용
                results.append({
                    'time': original_data[params['time_col']],
                    'user': original_data[params['user_col']],  # 원본 IP 주소
                    'process': original_data[params['proc_col']],  # 원본 프로세스 이름
                    'bytes': original_data[params['bytes_col']],
                    'raw_score': raw_score,  # 정규화 전 원본 점수
                    'anomaly_score': anomaly_score,  # 정규화된 최종 점수 (0~100)
                    'historical_influence': historical_influence,
                    'normalized_feature': normalized_feature,
                    'proc_normalized': proc_normalized,
                    'intensity': intensity.item(),
                    'is_anomaly': is_anomaly
                })

        # 모델 업데이트 (현재 사용자의 모델만 업데이트)
        if i % params['update_interval'] == 0:
            optimizer = optimizers[user_id]
            optimizer.zero_grad()
            loss = -torch.log(intensity).mean() * (1 + historical_influence)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 현재 사용자의 손실 기록
            all_losses[user_id].append(loss.item())

            # 모든 사용자의 평균 손실 표시
            all_recent_losses = []
            for u_id in unique_users:
                if all_losses[u_id]:  # 비어있지 않은 경우에만
                    all_recent_losses.extend(all_losses[u_id][-5:])

            if all_recent_losses:
                avg_loss = sum(all_recent_losses) / len(all_recent_losses)
                loss_text.text(f"현재 Loss: {loss.item():.4f} | 전체 평균 Loss: {avg_loss:.4f}")

        # 현재 사용자의 히스토리 업데이트
        histories[user_id].append({
            'time': current_time,
            'user_id': user_id_tensor,
            'proc_id': proc_id,
            'feature': feature
        })

        # 히스토리 크기 제한
        if len(histories[user_id]) > params['max_history_length']:
            histories[user_id] = histories[user_id][-params['max_history_length']:]

        # 체크포인트 저장 (모든 IP의 모델을 함께 저장)
        if i % params['checkpoint_interval'] == 0 and i > 0:
            checkpoint_path = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            save_dict = {
                'iteration': i,
                'histories': histories,
                'all_losses': all_losses,
                'recent_scores': recent_scores  # 점수 정규화에 필요한 최근 이상치 점수도 저장
            }

            # 각 IP별 모델과 옵티마이저 상태 저장
            for u_id in unique_users:
                save_dict[f'model_state_{u_id}'] = models[u_id].state_dict()
                save_dict[f'optimizer_state_{u_id}'] = optimizers[u_id].state_dict()

            torch.save(save_dict, checkpoint_path)

    progress_bar.empty()
    status_text.empty()
    loss_text.empty()

    # 모든 IP의 손실을 하나의 리스트로 결합 (시각화용)
    combined_losses = []
    for u_id in unique_users:
        combined_losses.extend(all_losses[u_id])

    # 첫 번째 모델을 반환 (호환성 위해)
    return models[unique_users[0]], pd.DataFrame(results), histories, combined_losses


# 결과 분석 및 시각화 함수
def analyze_and_visualize_results(results_df, losses):
    # 결과 분석
    total_events = len(results_df)
    total_anomalies = results_df['is_anomaly'].sum()
    anomaly_rate = (total_anomalies / total_events) * 100

    # 시각화
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 이상치 탐지 결과")
        st.metric("전체 이벤트 수", f"{total_events:,}")
        st.metric("탐지된 이상치 수", f"{int(total_anomalies):,}")
        st.metric("이상치 비율", f"{anomaly_rate:.2f}%")

        # 시계열 그래프
        fig_timeline = px.scatter(results_df,
                                  x='time',
                                  y='anomaly_score',
                                  color='is_anomaly',
                                  title='🕒 시간별 이상치 점수')
        st.plotly_chart(fig_timeline, use_container_width=True)

    with col2:
        # 사용자별 이상치 분포
        user_anomalies = results_df[results_df['is_anomaly']]['user'].value_counts()
        fig_users = px.bar(user_anomalies,
                           title='👤 사용자별 이상치 발생 횟수',
                           labels={'index': '사용자', 'value': '이상치 횟수'})
        st.plotly_chart(fig_users, use_container_width=True)

        # 학습 손실 그래프
        fig_loss = px.line(y=losses,
                           title='📉 학습 손실 추이',
                           labels={'index': '반복 횟수', 'value': '손실값'})
        st.plotly_chart(fig_loss, use_container_width=True)

    # 상세 분석 결과
    with st.expander("📋 상세 분석 결과 보기"):
        st.dataframe(results_df[results_df['is_anomaly']].sort_values('anomaly_score', ascending=False))


def main():
    # 사이드바 설정
    st.sidebar.title("✨ 설정 메뉴")

    # 파일 업로드 섹션
    st.sidebar.markdown("### 📁 파일 업로드")
    uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드해주세요! 😊", type=['csv'])

    # 메인 페이지 제목
    st.title("💝 IP 기반 시계열 이상치 탐지 시스템")
    st.markdown("#### 🌟 마크된 Hawkes Process를 사용한 IP별 이상치 탐지")

    if uploaded_file is not None:
        # 데이터 로드
        df = pd.read_csv(uploaded_file)

        # 사이드바에 파라미터 설정
        st.sidebar.markdown("### ⚙️ 컬럼 설정")
        time_col = st.sidebar.selectbox("시간 컬럼을 선택해주세요 ⏰", df.columns)
        user_col = st.sidebar.selectbox("사용자 컬럼을 선택해주세요 👤", df.columns)
        proc_col = st.sidebar.selectbox("프로세스 컬럼을 선택해주세요 🔄", df.columns)
        bytes_col = st.sidebar.selectbox("바이트 전송량 컬럼을 선택해주세요 📊", df.columns)

        st.sidebar.markdown("### 🎯 모델 파라미터")
        lr = st.sidebar.number_input("Learning Rate ✨", value=0.0005, format="%.4f")
        hidden_dim = st.sidebar.number_input("Hidden Dimension 🎈", value=128, min_value=32, max_value=512)
        time_window = st.sidebar.number_input("Time Window ⏱️", value=7200, min_value=1800)
        threshold = st.sidebar.number_input("Threshold 📊", value=8.0, format="%.1f")

        # 고급 설정
        with st.sidebar.expander("🔧 고급 설정"):
            warm_up_period = st.number_input("Warm-up Period", value=1000, min_value=100)
            update_interval = st.number_input("Update Interval", value=200, min_value=50)
            max_history_length = st.number_input("Max History Length", value=2000, min_value=500)
            checkpoint_interval = st.number_input("Checkpoint Interval", value=10000, min_value=1000)

        # 메인 영역에 데이터 미리보기 표시
        st.markdown("### 📊 데이터 미리보기")
        st.dataframe(df.head())

        # 실행 버튼

        if st.sidebar.button("✨ 이상치 탐지 시작!", help="클릭하면 이상치 탐지를 시작합니다"):
            try:
                with st.spinner("🔍 데이터 전처리 중..."):
                    # 원본 데이터 저장
                    original_df = df.copy()

                    # 전처리
                    df_processed, user_mapping, proc_mapping, user_encoder, proc_encoder, _ = preprocess_edr_data(
                        df=df,
                        user_col=user_col,
                        proc_col=proc_col,
                        time_col=time_col,
                        bytes_col=bytes_col
                    )

                # 모델 파라미터 설정
                params = {
                    'lr': lr,
                    'hidden_dim': hidden_dim,
                    'time_window': time_window,
                    'threshold': threshold,
                    'warm_up_period': warm_up_period,
                    'update_interval': update_interval,
                    'max_history_length': max_history_length,
                    'checkpoint_interval': checkpoint_interval,
                    'user_col': user_col,
                    'proc_col': proc_col,
                    'time_col': time_col,
                    'bytes_col': bytes_col
                }

                # 모델 학습 및 이상치 탐지 - 원본 데이터 전달 추가
                model, results_df, history, losses = edr_online_train_and_detect(
                    df_processed,
                    len(user_mapping),
                    len(proc_mapping),
                    user_encoder,
                    proc_encoder,
                    original_df,
                    **params
                )

                # 결과 분석 및 시각화
                analyze_and_visualize_results(results_df, losses)

                # 결과 다운로드 버튼
                st.download_button(
                    label="📥 결과 다운로드",
                    data=results_df.to_csv(index=False),
                    file_name="anomaly_detection_results.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"😢 오류가 발생했습니다: {str(e)}")

    else:
        # 파일이 업로드되지 않았을 때 표시할 메시지
        st.info("👈 왼쪽 사이드바에서 CSV 파일을 업로드해주세요! 🙂")

        # 사용 방법 안내
        st.markdown("""
        ### 📌 사용 방법
        1. 왼쪽 사이드바에서 CSV 파일을 업로드해주세요
        2. 필요한 컬럼들을 선택해주세요
        3. 모델 파라미터를 설정해주세요
        4. '이상치 탐지 시작' 버튼을 클릭하세요

        ### 🎯 주요 기능
        - IP 기반 시계열 데이터 분석
        - 실시간 이상치 탐지
        - 결과 시각화 및 다운로드
        """)


if __name__ == "__main__":
    main()
