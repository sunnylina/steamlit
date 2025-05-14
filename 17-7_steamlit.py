import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


def load_data():
    """사용자로부터 CSV 파일을 업로드받아 데이터프레임으로 변환합니다."""
    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # timestamp 컬럼이 있으면 datetime으로 변환
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            st.error(f"파일 로드 중 오류가 발생했습니다: {e}")
            return None
    else:
        return None


# Streamlit 앱 코드
def main():
    st.title('네트워크 트래픽 이상치 탐지 대시보드')
    st.write("CSV 파일을 업로드하여 네트워크 트래픽 이상치를 분석하세요.")

    # 데이터 로드
    df = load_data()

    # 데이터가 로드된 경우에만 분석 실행
    if df is not None:
        # 필요한 컬럼이 있는지 확인
        required_columns = ['user', 'process', 'anomaly', 'timestamp']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"필요한 컬럼이 누락되었습니다: {', '.join(missing_columns)}")
            st.write("CSV 파일에는 다음 컬럼이 포함되어야 합니다: user, process, anomaly, timestamp")
            return

        # 기본 통계 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 데이터 수", f"{len(df):,}")
        with col2:
            st.metric("이상치 수", f"{df['anomaly'].sum():,}")
        with col3:
            st.metric("이상치 비율", f"{df['anomaly'].mean():.2%}")

        # 시간에 따른 이상치 탐지 추이
        st.subheader('시간에 따른 이상치 발생 추이')
        try:
            time_fig = px.line(df.groupby(pd.Grouper(key='timestamp', freq='1h'))['anomaly'].mean().reset_index(),
                               x='timestamp', y='anomaly')
            st.plotly_chart(time_fig)
        except Exception as e:
            st.error(f"시간 그래프 생성 중 오류 발생: {e}")

        # User별 필터링 추가
        st.subheader('사용자별 프로세스 이상치 분석')

        # 전체 User 목록 가져오기
        all_users = df['user'].unique().tolist()

        # User 선택 위젯 추가
        selected_user = st.selectbox('분석할 사용자 선택:', ['전체 사용자'] + all_users)

        # 선택된 User로 데이터 필터링
        if selected_user == '전체 사용자':
            filtered_user_df = df
        else:
            filtered_user_df = df[df['user'] == selected_user]

        # 프로세스별 이상치 분석 (User 필터링 적용)
        process_counts = filtered_user_df.groupby('process')['anomaly'].agg(['count', 'mean']).reset_index()
        process_counts.columns = ['프로세스 ID', '발생 횟수', '이상치 비율']

        # 데이터가 존재하는 경우에만 그래프 생성
        if not process_counts.empty:
            process_fig = px.scatter(process_counts, x='발생 횟수', y='이상치 비율',
                                     hover_name='프로세스 ID', size='발생 횟수',
                                     log_x=True, title=f"{selected_user} 프로세스별 이상치 분석")
            st.plotly_chart(process_fig)
        else:
            st.info("선택한 사용자에 대한 데이터가 없습니다.")

        # User와 프로세스의 교차 분석 (히트맵)
        st.subheader('사용자-프로세스 교차 이상치 분석')

        try:
            cross_data = df.pivot_table(
                index='user',
                columns='process',
                values='anomaly',
                aggfunc='mean',
                fill_value=0
            ).reset_index()

            # 상위 10개 User만 표시 (성능 향상을 위해)
            top_users = df.groupby('user')['anomaly'].mean().nlargest(10).index.tolist()
            cross_data_filtered = cross_data[cross_data['user'].isin(top_users)]

            # 히트맵 그리기
            if not cross_data_filtered.empty and len(cross_data_filtered.columns) > 1:
                heatmap_data = cross_data_filtered.set_index('user')
                fig = px.imshow(heatmap_data,
                                labels=dict(x="프로세스", y="사용자", color="이상치 비율"),
                                title="사용자별-프로세스별 이상치 발생 비율")
                st.plotly_chart(fig)
            else:
                st.info("히트맵을 표시할 충분한 데이터가 없습니다.")
        except Exception as e:
            st.error(f"히트맵 생성 중 오류 발생: {e}")

        # 필터링 기능
        st.subheader('이상치 데이터 탐색')
        only_anomalies = st.checkbox('이상치만 보기')

        # User 필터링 유지
        if selected_user != '전체 사용자':
            display_df = filtered_user_df
        else:
            display_df = df

        # 이상치 필터링
        if only_anomalies:
            display_df = display_df[display_df['anomaly'] == 1]

        # 첫 100개 행 표시
        st.write(f"총 {len(display_df):,}개 행 중 첫 100개 행이 표시됩니다:")
        st.dataframe(display_df.head(100))

        # CSV 다운로드 버튼 추가
        st.download_button(
            label="필터링된 데이터 다운로드",
            data=display_df.to_csv(index=False).encode('utf-8'),
            file_name='filtered_data.csv',
            mime='text/csv',
        )
    else:
        # 샘플 데이터 형식 안내
        st.info("데이터를 분석하려면 CSV 파일을 업로드하세요.")
        st.write("CSV 파일에는 다음 컬럼이 포함되어야 합니다:")
        sample_data = pd.DataFrame({
            'timestamp': ['2025-05-14 10:00:00', '2025-05-14 10:01:00'],
            'user': ['user123', 'user456'],
            'process': ['chrome.exe', 'explorer.exe'],
            'anomaly': [0, 1]
        })
        st.dataframe(sample_data)


if __name__ == "__main__":
    main()
