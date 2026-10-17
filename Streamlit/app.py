import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity
)

# ============================================================
# IMAGE DENOISING DASHBOARD
# ============================================================
# Chức năng:
# 1. So sánh 3 mức nhiễu bằng đúng 9 hình
# 2. Kết quả thực nghiệm 42 trường hợp
# 3. Khi chọn điều kiện nhiễu + bộ lọc:
#    hiển thị đầy đủ MSE, PSNR, SSIM
#
# Không có chức năng "Demo một ảnh".
# Không sử dụng thời gian xử lý.
# ============================================================


# ============================================================
# 1. CONFIG
# ============================================================

st.set_page_config(
    page_title="Image Denoising Dashboard",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Denoising Dashboard")
st.caption(
    "Nghiên cứu và so sánh hiệu quả của các phương pháp khử nhiễu ảnh "
    "dưới các mức độ nhiễu khác nhau."
)


# ============================================================
# 2. PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
GRAY_DIR = DATA_DIR / "processed" / "grayscale"

RESULT_DIR = BASE_DIR / "results"

DETAIL_CSV = RESULT_DIR / "01_Ket_qua_chi_tiet_6300_luot.csv"
SUMMARY_CSV = RESULT_DIR / "02_Ket_qua_tong_hop_42_truong_hop.csv"
BEST_CSV = RESULT_DIR / "03_Bo_loc_tot_nhat_theo_muc_nhieu.csv"


# ============================================================
# 3. CONSTANTS
# ============================================================

NOISE_TYPES = [
    "Gaussian Noise",
    "Salt-and-Pepper Noise"
]

NOISE_LEVELS = [10, 20, 30]

FILTERS = [
    "Mean 3x3",
    "Mean 5x5",
    "Gaussian 3x3",
    "Gaussian 5x5",
    "Median 3x3",
    "Median 5x5",
    "Bilateral"
]


# ============================================================
# 4. NOISE FUNCTIONS
# ============================================================

def add_gaussian_noise(image, level_percent, seed=42):
    """
    Gaussian Noise:
    10% -> sigma = 25.5
    20% -> sigma = 51
    30% -> sigma = 76.5
    """

    rng = np.random.default_rng(seed)

    sigma = (level_percent / 100.0) * 255.0

    noise = rng.normal(
        loc=0,
        scale=sigma,
        size=image.shape
    )

    noisy = image.astype(np.float32) + noise

    noisy = np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)

    return noisy


def add_salt_pepper_noise(image, level_percent, seed=42):
    """
    Salt-and-Pepper Noise:
    level_percent = tỷ lệ pixel bị thay đổi.

    Một nửa thành 0 (Pepper)
    Một nửa thành 255 (Salt)
    """

    rng = np.random.default_rng(seed)

    noisy = image.copy()

    total_pixels = image.shape[0] * image.shape[1]

    number_of_noisy_pixels = int(
        total_pixels * level_percent / 100
    )

    if number_of_noisy_pixels <= 0:
        return noisy

    indices = rng.choice(
        total_pixels,
        size=number_of_noisy_pixels,
        replace=False
    )

    salt_count = number_of_noisy_pixels // 2

    salt_indices = indices[:salt_count]
    pepper_indices = indices[salt_count:]

    noisy_flat = noisy.reshape(-1)

    noisy_flat[salt_indices] = 255
    noisy_flat[pepper_indices] = 0

    return noisy


def create_noisy_image(image, noise_type, level):
    if noise_type == "Gaussian Noise":
        return add_gaussian_noise(
            image,
            level,
            seed=42
        )

    return add_salt_pepper_noise(
        image,
        level,
        seed=42
    )


# ============================================================
# 5. DENOISING FUNCTIONS
# ============================================================

def denoise_image(image, filter_name):

    if filter_name == "Mean 3x3":
        return cv2.blur(
            image,
            (3, 3)
        )

    elif filter_name == "Mean 5x5":
        return cv2.blur(
            image,
            (5, 5)
        )

    elif filter_name == "Gaussian 3x3":
        return cv2.GaussianBlur(
            image,
            (3, 3),
            0
        )

    elif filter_name == "Gaussian 5x5":
        return cv2.GaussianBlur(
            image,
            (5, 5),
            0
        )

    elif filter_name == "Median 3x3":
        return cv2.medianBlur(
            image,
            3
        )

    elif filter_name == "Median 5x5":
        return cv2.medianBlur(
            image,
            5
        )

    elif filter_name == "Bilateral":
        return cv2.bilateralFilter(
            image,
            9,
            75,
            75
        )

    return image


# ============================================================
# 6. METRICS
# ============================================================

def calculate_metrics(original, denoised):

    mse = mean_squared_error(
        original,
        denoised
    )

    psnr = peak_signal_noise_ratio(
        original,
        denoised,
        data_range=255
    )

    ssim = structural_similarity(
        original,
        denoised,
        data_range=255
    )

    return mse, psnr, ssim


# ============================================================
# 7. READ / UPLOAD IMAGE
# ============================================================

def uploaded_to_gray(uploaded_file):
    image = Image.open(uploaded_file).convert("L")
    image = np.array(image)
    image = cv2.resize(
        image,
        (256, 256),
        interpolation=cv2.INTER_AREA
    )
    return image


# ============================================================
# 9. LOAD RESULTS
# ============================================================

@st.cache_data
def load_csv(path):

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


detail_df = load_csv(DETAIL_CSV)
summary_df = load_csv(SUMMARY_CSV)
best_df = load_csv(BEST_CSV)


# ============================================================
# 10. HELPER: FIND COLUMN
# ============================================================

def find_column(df, candidates):

    if df is None:
        return None

    # Ưu tiên khớp chính xác
    for candidate in candidates:

        for col in df.columns:

            if str(col).strip().lower() == candidate.lower():
                return col

    # Sau đó tìm theo từ khóa
    for candidate in candidates:

        candidate_lower = candidate.lower()

        for col in df.columns:

            if candidate_lower in str(col).lower():
                return col

    return None


def find_noise_column(df):

    return find_column(
        df,
        [
            "Noise",
            "Noise Type",
            "Loai Nhieu",
            "Loại nhiễu",
            "Nhiễu"
        ]
    )


def find_filter_column(df):

    return find_column(
        df,
        [
            "Filter",
            "Filter Name",
            "Bo Loc",
            "Bộ lọc"
        ]
    )


def find_level_column(df):

    return find_column(
        df,
        [
            "Level",
            "Noise Level",
            "Muc Nhieu",
            "Mức nhiễu"
        ]
    )


def find_metric_column(df, metric):

    mapping = {
        "MSE": [
            "MSE",
            "MSE_TB",
            "MSE TB",
            "MSE trung bình"
        ],
        "PSNR": [
            "PSNR",
            "PSNR_TB",
            "PSNR TB",
            "PSNR trung bình"
        ],
        "SSIM": [
            "SSIM",
            "SSIM_TB",
            "SSIM TB",
            "SSIM trung bình"
        ]
    }

    return find_column(
        df,
        mapping[metric]
    )


# ============================================================
# 11. FIND EXPERIMENT RESULT
# ============================================================

def get_experiment_result(
    noise_type,
    level,
    filter_name
):

    if summary_df is None:
        return None

    df = summary_df.copy()

    noise_col = find_noise_column(df)
    filter_col = find_filter_column(df)

    if noise_col is None or filter_col is None:
        return None

    # Các cách viết có thể có trong CSV
    noise_aliases = []

    if noise_type == "Gaussian Noise":
        noise_aliases = [
            "Gaussian Noise",
            "Gaussian"
        ]
    else:
        noise_aliases = [
            "Salt-and-Pepper Noise",
            "Salt-Pepper Noise",
            "Salt-Pepper",
            "Salt and Pepper",
            "Salt-and-Pepper"
        ]

    # Tìm chính xác Filter + Noise
    result = pd.DataFrame()

    for noise_name in noise_aliases:

        target_noise = f"{noise_name} {level}%"

        temp = df[
            (
                df[noise_col]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                target_noise.lower()
            )
            &
            (
                df[filter_col]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                filter_name.lower()
            )
        ]

        if len(temp) > 0:
            result = temp
            break

    # Nếu CSV tách Noise và Level thành 2 cột
    if len(result) == 0:

        level_col = find_level_column(df)

        if level_col is not None:

            for noise_name in noise_aliases:

                temp = df[
                    df[noise_col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .eq(noise_name.lower())
                    &
                    df[level_col]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .str.strip()
                    .eq(str(level))
                    &
                    df[filter_col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .eq(filter_name.lower())
                ]

                if len(temp) > 0:
                    result = temp
                    break

    # Fallback contains
    if len(result) == 0:

        noise_text = noise_type.replace(" Noise", "")

        result = df[
            df[noise_col]
            .astype(str)
            .str.contains(
                noise_text,
                case=False,
                na=False
            )
            &
            df[noise_col]
            .astype(str)
            .str.contains(
                str(level),
                case=False,
                na=False
            )
            &
            df[filter_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(filter_name.lower())
        ]

    if len(result) == 0:
        return None

    return result.iloc[0]


# ============================================================
# 12. DISPLAY EXPERIMENT METRICS
# ============================================================

def display_experiment_metrics(
    noise_type,
    level,
    filter_name
):

    result = get_experiment_result(
        noise_type,
        level,
        filter_name
    )

    st.markdown(
        f"#### {noise_type} – {level}% – {filter_name}"
    )

    m1, m2, m3 = st.columns(3)

    if result is not None:

        mse_col = find_metric_column(
            summary_df,
            "MSE"
        )

        psnr_col = find_metric_column(
            summary_df,
            "PSNR"
        )

        ssim_col = find_metric_column(
            summary_df,
            "SSIM"
        )

        # MSE
        with m1:

            if mse_col is not None:
                st.metric(
                    "MSE",
                    f"{float(result[mse_col]):.4f}"
                )
            else:
                st.metric(
                    "MSE",
                    "N/A"
                )

        # PSNR
        with m2:

            if psnr_col is not None:
                st.metric(
                    "PSNR",
                    f"{float(result[psnr_col]):.4f} dB"
                )
            else:
                st.metric(
                    "PSNR",
                    "N/A"
                )

        # SSIM
        with m3:

            if ssim_col is not None:
                st.metric(
                    "SSIM",
                    f"{float(result[ssim_col]):.4f}"
                )
            else:
                st.metric(
                    "SSIM",
                    "N/A"
                )

    else:

        for col in [m1, m2, m3]:
            with col:
                st.metric(
                    "Kết quả",
                    "Không tìm thấy"
                )


# ============================================================
# 13. DISPLAY 3x3 MATRIX FOR ONE FILTER
# ============================================================

def gaussian_sigma(level):
    return (level / 100.0) * 255.0


def display_filter_matrix(
    original,
    noise_type,
    filter_name
):
    st.subheader(f"🔹 {filter_name}")

    h1, h2, h3 = st.columns(3)

    with h1:
        st.markdown("**ẢNH GỐC**")
    with h2:
        st.markdown("**ẢNH NHIỄU**")
    with h3:
        st.markdown("**SAU KHỬ NHIỄU**")

    # Chỉ hiển thị 9 hình:
    # 3 mức nhiễu × (Ảnh gốc + Ảnh nhiễu + Ảnh sau khử nhiễu)
    for level in NOISE_LEVELS:

        noisy = create_noisy_image(
            original,
            noise_type,
            level
        )

        denoised = denoise_image(
            noisy,
            filter_name
        )

        c1, c2, c3 = st.columns(3)

        if noise_type == "Gaussian Noise":
            noise_caption = (
                f"Gaussian {level}% — σ = "
                f"{gaussian_sigma(level):.1f}"
            )
            denoise_caption = (
                f"{filter_name} — {level}% "
                f"(σ = {gaussian_sigma(level):.1f})"
            )
        else:
            noise_caption = f"Salt-and-Pepper {level}%"
            denoise_caption = f"{filter_name} — {level}%"

        with c1:
            st.image(
                original,
                caption=f"Ảnh gốc — mức {level}%",
                use_container_width=True
            )

        with c2:
            st.image(
                noisy,
                caption=noise_caption,
                use_container_width=True
            )

        with c3:
            st.image(
                denoised,
                caption=denoise_caption,
                use_container_width=True
            )

        # Hiển thị 3 chỉ số tương ứng với trường hợp này
        display_experiment_metrics(
            noise_type,
            level,
            filter_name
        )

        st.divider()


# ============================================================
# 14. SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Cấu hình")

mode = st.sidebar.radio(
    "Chế độ",
    [
        "So sánh 3 mức nhiễu",
        "Kết quả thực nghiệm"
    ]
)


# ============================================================
# 15. MODE: SO SÁNH 3 MỨC NHIỄU
# ============================================================

if mode == "So sánh 3 mức nhiễu":

    st.sidebar.subheader("📷 Chọn ảnh")

    uploaded_file = st.sidebar.file_uploader(
        "Upload ảnh",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp"
        ]
    )

    if uploaded_file is None:
        st.info(
            "📌 Hãy upload một ảnh để bắt đầu."
        )
        st.stop()

    original = uploaded_to_gray(uploaded_file)
    image_name = uploaded_file.name

    if original is None:
        st.error("Không thể đọc ảnh.")
        st.stop()

    st.sidebar.success(
        f"Đã chọn: {image_name}"
    )

    noise_type = st.sidebar.selectbox(
        "Loại nhiễu",
        NOISE_TYPES
    )

    # Chỉ chọn 1 bộ lọc để màn hình hiển thị đúng 9 hình
    filter_name = st.sidebar.selectbox(
        "Bộ lọc",
        FILTERS
    )

    st.header(
        f"📊 So sánh {noise_type} ở 3 mức độ"
    )

    st.write(
        f"Ảnh: **{image_name}** — Bộ lọc: **{filter_name}**"
    )

    st.info(
        "Màn hình hiển thị đúng **9 hình**: "
        "3 mức nhiễu × 3 trạng thái (ảnh gốc, ảnh nhiễu, ảnh sau khử nhiễu)."
    )

    display_filter_matrix(
        original,
        noise_type,
        filter_name
    )


# ============================================================
# 16. MODE: KẾT QUẢ THỰC NGHIỆM
# ============================================================

else:

    st.header(
        "📊 Kết quả thực nghiệm – 42 trường hợp"
    )

    st.write(
        "2 loại nhiễu × 3 mức nhiễu × 7 cấu hình bộ lọc "
        "= 42 trường hợp."
    )

    # --------------------------------------------------------
    # Check CSV
    # --------------------------------------------------------

    if summary_df is None:

        st.warning(
            "⚠️ Chưa tìm thấy file:\n\n"
            f"`{SUMMARY_CSV}`"
        )

        st.info(
            "Hãy đặt file "
            "`02_Ket_qua_tong_hop_42_truong_hop.csv` "
            "vào thư mục `results`."
        )

        st.stop()

    df = summary_df.copy()

    noise_col = find_noise_column(df)
    filter_col = find_filter_column(df)

    # --------------------------------------------------------
    # SIDEBAR FILTER
    # --------------------------------------------------------

    st.sidebar.subheader(
        "🔎 Lọc kết quả thực nghiệm"
    )

    # Điều kiện nhiễu
    if noise_col is not None:

        noise_options = (
            ["Tất cả"]
            +
            sorted(
                df[noise_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

        selected_noise_condition = st.sidebar.selectbox(
            "Điều kiện nhiễu",
            noise_options
        )

    else:

        selected_noise_condition = "Tất cả"

    # Bộ lọc
    if filter_col is not None:

        filter_options = (
            ["Tất cả"]
            +
            sorted(
                df[filter_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

        selected_filter_condition = st.sidebar.selectbox(
            "Bộ lọc",
            filter_options
        )

    else:

        selected_filter_condition = "Tất cả"

    # --------------------------------------------------------
    # APPLY FILTER
    # --------------------------------------------------------

    filtered_df = df.copy()

    if (
        noise_col is not None
        and selected_noise_condition != "Tất cả"
    ):

        filtered_df = filtered_df[
            filtered_df[noise_col]
            .astype(str)
            .eq(selected_noise_condition)
        ]

    if (
        filter_col is not None
        and selected_filter_condition != "Tất cả"
    ):

        filtered_df = filtered_df[
            filtered_df[filter_col]
            .astype(str)
            .eq(selected_filter_condition)
        ]

    # --------------------------------------------------------
    # SELECTED CONDITION RESULT
    # --------------------------------------------------------

    if (
        selected_noise_condition != "Tất cả"
        and selected_filter_condition != "Tất cả"
    ):

        st.subheader(
            "🎯 Kết quả theo điều kiện đã chọn"
        )

        # Tìm 3 metric
        mse_col = find_metric_column(
            filtered_df,
            "MSE"
        )

        psnr_col = find_metric_column(
            filtered_df,
            "PSNR"
        )

        ssim_col = find_metric_column(
            filtered_df,
            "SSIM"
        )

        if len(filtered_df) > 0:

            row = filtered_df.iloc[0]

            m1, m2, m3 = st.columns(3)

            with m1:

                if mse_col is not None:

                    st.metric(
                        "MSE",
                        f"{float(row[mse_col]):.4f}"
                    )

                else:

                    st.metric(
                        "MSE",
                        "N/A"
                    )

            with m2:

                if psnr_col is not None:

                    st.metric(
                        "PSNR",
                        f"{float(row[psnr_col]):.4f} dB"
                    )

                else:

                    st.metric(
                        "PSNR",
                        "N/A"
                    )

            with m3:

                if ssim_col is not None:

                    st.metric(
                        "SSIM",
                        f"{float(row[ssim_col]):.4f}"
                    )

                else:

                    st.metric(
                        "SSIM",
                        "N/A"
                    )

        else:

            st.warning(
                "Không tìm thấy kết quả cho điều kiện đã chọn."
            )

    else:

        st.info(
            "Chọn đồng thời **Điều kiện nhiễu** và "
            "**Bộ lọc** để xem đầy đủ MSE, PSNR và SSIM."
        )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader(
        "📋 Bảng kết quả thực nghiệm"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Đang hiển thị **{len(filtered_df)}** trường hợp."
    )

    # --------------------------------------------------------
    # BEST FILTER
    # --------------------------------------------------------

    if best_df is not None:

        st.subheader(
            "🏆 Bộ lọc tốt nhất theo mức nhiễu"
        )

        st.dataframe(
            best_df,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # DETAIL 6300
    # --------------------------------------------------------

    if detail_df is not None:

        with st.expander(
            "📄 Xem kết quả chi tiết 6.300 lượt đánh giá"
        ):

            st.write(
                f"Tổng số lượt đánh giá: "
                f"**{len(detail_df):,}**"
            )

            st.dataframe(
                detail_df,
                use_container_width=True,
                height=500,
                hide_index=True
            )


# ============================================================
# 17. FOOTER
# ============================================================

st.divider()

st.caption(
    "Digital Image Processing Project | "
    "BSDS500 | Gaussian Noise | Salt-and-Pepper Noise | "
    "Mean | Gaussian | Median | Bilateral | "
    "MSE | PSNR | SSIM"
)