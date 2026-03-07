import cv2
import numpy as np
import serial
import time
import keyboard
import struct
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# ================== 配置参数 ==================
# 串口配置
SERIAL_PORT = 'COM3'  # 串口号（根据实际修改）
BAUDRATE = 115200  # 必须与下位机一致

#标靶基础参数
REAL_WIDTH = 5.0  # 标靶真实边长(cm)，精确到小数点后1位
TARGET_SIZE = 5.0  # 标靶边长(cm)
MEASURE_RANGE = [40, 60]  # 测量范围(cm)

# 多距离标定K值 格式：{距离(cm): 像素宽度(px), ...}，建议标定40/45/50/55/60cm
CALIB_DATA = {
    40: 90.0,  # 示例：40cm处像素宽度90px
    45: 80.0,  # 45cm处80px
    50: 72.0,  # 50cm处72px
    55: 65.0,  # 55cm处65px
    60: 60.0  # 60cm处60px
}
# 拟合K值曲线（多项式拟合，替代单一K值）
calib_distances = np.array(list(CALIB_DATA.keys()))
calib_pixel_widths = np.array(list(CALIB_DATA.values()))
# 拟合K值：K = a*distance + b（线性拟合）
fit_coeff = np.polyfit(calib_distances, calib_pixel_widths * calib_distances / REAL_WIDTH, 1)
K_FUNC = lambda d: fit_coeff[0] * d + fit_coeff[1]  # 不同距离的K值函数
# 滤波配置
PIXEL_WIDTH_BUFFER = deque(maxlen=5)  # 像素宽度滑动窗口（5帧平滑）
SMOOTH_WINDOW_SIZE = 5  # 平滑窗口大小

# 摄像头配置
CAMERA_ID = 0
FRAME_SIZE = (640, 480)  # 摄像头分辨率
IMG_CENTER_X = FRAME_SIZE[0] // 2  # 画面中心X
IMG_CENTER_Y = FRAME_SIZE[1] // 2  # 画面中心Y

# 标靶识别配置（正方形标靶）
MIN_CONTOUR_AREA = 30  # 最小有效标靶面积（像素）
SQUARE_ASPECT_RATIO = (0.5, 1.5)  # 正方形宽高比范围
# HSV颜色阈值（标靶颜色，根据实际标靶调整）
LOWER_TARGET = np.array([0, 0, 0])
UPPER_TARGET = np.array([180, 255, 180])

# 显示配置
CROSS_COLOR = (0, 255, 0)  # 中心十字颜色（BGR）
FPS_POS = (10, 30)  # 帧率显示位置
prev_time = cv2.getTickCount()

# 目标丢失计数器
lost_counter = 0
MAX_LOST_FRAMES = 10
# 用于存储当前任务
current_task = 0
exit_flag = False
# ================== 函数定义 ==================
def adaptive_threshold(img_gray):
    """自适应二值化：适配不同光线，精准提取黑色标靶"""
    # 高斯自适应二值化，blockSize必须为奇数，C为常数（调整对比度）
    thresh = cv2.adaptiveThreshold(
        img_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,  # 邻域大小，越大越平滑
        C=2  # 常数，越小越灵敏
    )
    # 形态学闭运算：消除小孔洞，填充标靶内部缝隙
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh

def is_square_contour_1(contour, min_area=20 * 20, aspect_ratio_range=(0.9, 1.1)):
    """精准筛选正方形轮廓：几何特征+近似度+凸包检测"""
    # 1. 面积筛选
    area = cv2.contourArea(contour)
    if area < min_area:
        return False, None

    # 2. 轮廓近似（多边形拟合）
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)  # 0.02为近似精度

    # 3. 正方形特征：4个顶点+凸包+宽高比接近1
    if len(approx) == 4 and cv2.isContourConvex(approx):
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
            # 计算最小外接矩形（消除透视变形影响）
            rect = cv2.minAreaRect(contour)
            (cx, cy), (w_min, h_min), angle = rect
            # 正方形的最小外接矩形宽高比也应接近1
            if 0.9 <= (w_min / h_min) <= 1.1:
                return True, (x, y, w, h, cx, cy)

    return False, None

def distance_calculation(pixel_width, current_k):
    """高精度距离计算：基于拟合K值+误差补偿"""
    distance = (current_k * REAL_WIDTH) / pixel_width
    # 40-60cm范围误差补偿（根据实际标定微调）
    if 40 <= distance <= 60:
        # 示例补偿：根据实测偏差调整，比如50cm处偏+0.3cm，就减0.3
        distance -= 0.2  # 可根据实际标定结果微调
    return round(distance, 1)


def send_distance_via_serial(avg_distance):
    # 获取当前时间并格式化为 HH:MM:SS 字符串
    current_time_str = time.strftime("%H:%M:%S", time.localtime())
    print(f"当前时间: {current_time_str}")
    # 将时间字符串编码为字节
    time_bytes = current_time_str.encode('utf-8')  # 编码为UTF-8字节流
    # 发送平均值和时间字符串到单片机
    try:
        # 将平均值打包为4字节浮点数
        avg_bytes = struct.pack('f', avg_distance)

        # 构造协议帧：[0xA5][0xC5][平均值(4字节)][时间字符串(最多8字节)]
        data = bytes([0xA5, 0xC5]) + avg_bytes + time_bytes
        ser.write(data)
        print("数据已发送到单片机")
        return True
    except Exception as e:
        print(f"串口发送失败: {e}")
        return False

def monocular_distance_measurement():
    global exit_flag, K_FUNC, current_task
    #结果存储取平均值
    distance_list = []
    MAX_DEVIATION = 0.8  # 最大允许偏差
    comfirm_flag = False
    #初始化摄像头
    cap = initialize_camera(CAMERA_ID, FRAME_SIZE)
    #摄像头循环输出图像测距
    while True:
        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败！")
            break
        frame_show = frame.copy()
        #高精度图像预处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 高斯模糊（核大小5x5，sigmaX=1.5，适配笔记本摄像头）
        blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
        # 自适应二值化+形态学操作
        thresh = adaptive_threshold(blur)
        #精准轮廓提取与筛选
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        target_found = False
        pixel_width = 0
        distance = 0.0
        center_x, center_y = 0, 0

        if contours:
            # 按面积排序，优先处理大轮廓
            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours_sorted[:3]:  # 只处理前3个最大轮廓，提升效率
                is_square, square_info = is_square_contour_1(cnt)
                if is_square:
                    x, y, w, h, cx, cy = square_info
                    pixel_width = (w + h) / 2  # 取宽高均值，减少透视误差
                    center_x, center_y = cx, cy
                    target_found = True
                    break

        #高精度测距计算
        if target_found:
            # 像素宽度滑动平均滤波
            PIXEL_WIDTH_BUFFER.append(pixel_width)
            if len(PIXEL_WIDTH_BUFFER) >= SMOOTH_WINDOW_SIZE:
                pixel_width_smooth = np.mean(PIXEL_WIDTH_BUFFER)
            else:
                pixel_width_smooth = pixel_width

            # 预估当前距离，计算适配的K值（迭代优化）
            distance_pre = (K_FUNC(50) * REAL_WIDTH) / pixel_width_smooth  # 初始用50cm的K值
            # 限制K值范围在40-60cm之间
            distance_pre = np.clip(distance_pre, MEASURE_RANGE[0], MEASURE_RANGE[1])
            current_k = K_FUNC(distance_pre)

            # 最终距离计算（带误差补偿）
            distance = distance_calculation(pixel_width_smooth, current_k)
            if len(distance_list) < 5:
                distance_list.append(distance)
            else:
                distance_list.pop(0)
                distance_list.append(distance)

            #高精度标注
            # 绘制最小外接矩形（更贴合实际正方形）
            cv2.rectangle(frame_show, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            # 绘制标靶中心（红色实心圆）
            cv2.circle(frame_show, (int(center_x), int(center_y)), 4, (0, 0, 255), -1)
            # 标注高精度信息
            frame_show = put_chinese_text(frame_show, f"像素宽度(平滑): {pixel_width_smooth:.1f} px", (10, 40),
                                          font_path="simhei.ttf", font_size=30, color=(255, 0, 0))
            frame_show = put_chinese_text(frame_show, f"当前K值: {current_k:.1f}", (10, 80),
                                          font_path="simhei.ttf", font_size=30, color=(0, 255, 255))
            frame_show = put_chinese_text(frame_show, f"测距结果: {distance} cm", (10, 120),
                                          font_path="simhei.ttf", font_size=40, color=(0, 0, 255))
            # 打印高精度日志
            print(f"【高精度测距】距离: {distance}cm | 像素宽度: {pixel_width_smooth:.1f}px | K值: {current_k:.1f}")
        else:
            # 未找到标靶提示
            frame_show = put_chinese_text(frame_show, "未识别到标靶！请调整标靶位置", (10, 40),
                                          font_path="simhei.ttf", font_size=30, color=(0, 0, 255))
            # 清空像素宽度缓存
            PIXEL_WIDTH_BUFFER.clear()
        # 显示画面（缩放至全屏，方便观察）
        cv2.imshow("High-Precision Monocular Distance Measurement (40-60cm 1cm)", frame_show)
        if comfirm_flag  and len(distance_list) >= 5:
            max_val = max(distance_list)
            min_val = min(distance_list)
            deviation = max_val - min_val
            if deviation < MAX_DEVIATION:
                # 计算平均值
                avg_distance = sum(distance_list) / len(distance_list)
                # 发送数据并通过串口
                if send_distance_via_serial(avg_distance):
                    print("发送成功，退出测量循环")
                    current_task = 0
                    break
            else:
                print(f"偏差过大，当前偏差: {deviation}")
        # 按键操作
        if exit_flag:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite("high_precision_calib.jpg", frame)
            print("标定帧已保存：high_precision_calib.jpg\n")
        elif key == ord('c'):
            # 重新标定K值（输入当前距离和像素宽度）
            current_dist = float(input("请输入当前标靶实际距离(cm)："))
            current_px = float(input("请输入当前标靶像素宽度(px)："))
            CALIB_DATA[current_dist] = current_px
            # 重新拟合K值曲线
            calib_distances = np.array(list(CALIB_DATA.keys()))
            calib_pixel_widths = np.array(list(CALIB_DATA.values()))
            fit_coeff = np.polyfit(calib_distances, calib_pixel_widths * calib_distances / REAL_WIDTH, 1)
            K_FUNC = lambda d: fit_coeff[0] * d + fit_coeff[1]
            print(f"重新标定完成！新K值函数：K = {fit_coeff[0]:.2f}*d + {fit_coeff[1]:.2f}\n")
        elif key == ord('q'):
            comfirm_flag = True
            print(f"开始测量，请保持标靶在画面正中间\n")
    # 释放资源
    exit_flag = False
    cap.release()
    cv2.destroyAllWindows()
    print("\n高精度测距程序已退出！")

def draw_center_cross(frame, arm_length=15, thickness=2):
    """在画面中心绘制校准十字"""
    cv2.line(frame, (IMG_CENTER_X - arm_length, IMG_CENTER_Y),
             (IMG_CENTER_X + arm_length, IMG_CENTER_Y), CROSS_COLOR, thickness)
    cv2.line(frame, (IMG_CENTER_X, IMG_CENTER_Y - arm_length),
             (IMG_CENTER_X, IMG_CENTER_Y + arm_length), CROSS_COLOR, thickness)

def get_fps():
    """计算实时帧率"""
    global prev_time
    current_time = cv2.getTickCount()
    delta_time = (current_time - prev_time) / cv2.getTickFrequency()
    fps = 1 / delta_time if delta_time > 0 else 0
    prev_time = current_time
    return int(fps)

def is_square_contour(contour):
    """判断轮廓是否为正方形"""
    # 计算轮廓外接矩形
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    # 宽高比在0.8~1.2之间，且面积达标
    if (SQUARE_ASPECT_RATIO[0] <= aspect_ratio <= SQUARE_ASPECT_RATIO[1] and
            cv2.contourArea(contour) > MIN_CONTOUR_AREA):
        # 轮廓近似（确保是四边形）
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        return len(approx) == 4
    return False

def send_servo_command(target_cx, target_cy):
    """发送舵机位置指令"""
    try:
        # 协议帧：[0xA4][0xC4][XL][XH][YL][YH]
        xl = target_cx & 0xFF
        xh = (target_cx >> 8) & 0xFF
        yl = target_cy & 0xFF
        yh = (target_cy >> 8) & 0xFF
        data = bytes([0xA4, 0xC4, xl, xh, yl, yh])
        ser.write(data)
        return True
    except Exception as e:
        print(f"串口发送失败: {e}")
        return False

def target_alignment():
    global exit_flag
    #初始化摄像头
    cap = initialize_camera(CAMERA_ID, FRAME_SIZE)
    comfirm_flag = False
    #摄像头循环输出图像测距
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频流读取失败，重试...")
            time.sleep(0.1)
            continue
        # 初始化帧状态
        target_found = False
        target_cx, target_cy = 0, 0
        lost_counter = 0

        frame = cv2.resize(frame, FRAME_SIZE)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 获取图像尺寸
        height, width = frame.shape[:2]

        # 计算中间区域的坐标
        roi_size = 300
        start_x = (width - roi_size) // 2
        start_y = (height - roi_size) // 2
        end_x = start_x + roi_size
        end_y = start_y + roi_size

        # 颜色分割
        mask_full = cv2.inRange(hsv, LOWER_TARGET, UPPER_TARGET)
        kernel = np.ones((5, 5), np.uint8)
        mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, kernel)  # 开运算去噪
        mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel)  # 闭运算填充
        # 只保留中间 200x200 区域的 mask，其他区域设为黑色
        mask = np.zeros_like(mask_full)
        mask[start_y:end_y, start_x:end_x] = mask_full[start_y:end_y, start_x:end_x]
        # 绘制 ROI 区域可视化（绿色矩形框，用于调试）
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        #轮廓检测与正方形筛选
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 筛选最大的正方形轮廓
            square_contours = [cnt for cnt in contours if is_square_contour(cnt)]
            if square_contours:
                max_contour = max(square_contours, key=cv2.contourArea)
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    # 计算标靶中心坐标
                    target_cx = int(M["m10"] / M["m00"])
                    target_cy = int(M["m01"] / M["m00"])
                    target_found = True
                    #绘制标靶信息
                    cv2.drawContours(frame, [max_contour], -1, (0, 0, 255), 2)  # 标靶轮廓
                    cv2.circle(frame, (target_cx, target_cy), 5, (255, 0, 0), -1)  # 标靶中心
                    cv2.putText(frame, f"Target ({target_cx},{target_cy})",
                                (target_cx + 10, target_cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        #目标丢失处理
        if not target_found:
            lost_counter += 1
            if lost_counter > MAX_LOST_FRAMES:
                comfirm_flag = False
                cv2.putText(frame, "TARGET LOST", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(f"请把舵机移动到画面正中间，并确认标靶已进入画面，按q开始执行程序")
        #发送舵机指令
        if comfirm_flag :
            if ser and ser.is_open:
                send_servo_command(target_cx, target_cy)
        #绘制辅助信息
        draw_center_cross(frame)  # 画面中心十字
        fps = get_fps()
        cv2.putText(frame, f"FPS: {fps}", FPS_POS,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        #显示画面
        cv2.imshow('Target Tracking & Servo Control', frame)
        cv2.imshow('Mask', mask)
        if exit_flag:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            comfirm_flag =  True
    exit_flag = False
    cap.release()
    cv2.destroyAllWindows()
    print("标靶对正程序已退出！")
def on_up():
    try:
        # 协议帧：[0xA3][0xC3][按键]
        data = bytes([0xA3, 0xC3,0xA1])
        ser.write(data)
    except Exception as e:
        print(f"串口发送失败: {e}")
    finally:
        print("Up key pressed!")

def on_down():
    try:
        # 协议帧：[0xA3][0xC3][按键]
        data = bytes([0xA3, 0xC3, 0xA2])
        ser.write(data)
    except Exception as e:
        print(f"串口发送失败: {e}")
    finally:
        print("Down key pressed!")

def on_enter():
    try:
        # 协议帧：[0xA3][0xC3][按键]
        data = bytes([0xA3, 0xC3, 0xA3])
        ser.write(data)
    except Exception as e:
        print(f"串口发送失败: {e}")
    finally:
        print("Enter key pressed!")

def on_ctrl_c():
    global current_task,exit_flag
    try:
        # 协议帧：[0xA3][0xC3][按键]
        data = bytes([0xA3, 0xC3, 0xA4])
        ser.write(data)
        exit_flag = True
    except Exception as e:
        print(f"串口发送失败: {e}")
    finally:
        current_task = 0
        print("Ctrl+C pressed!")

def receive_serial_command(ser):
    global current_task
    try:
        if ser.in_waiting > 0:
            data = ser.read(ser.in_waiting)
            print(f"接收到串口数据: {data}")
            # 根据协议解析数据[0xA4][0xC4][指令]
            if len(data) >= 3 and data[0] == 0xA4 and data[1] == 0xC4:
                command = data[2]  # 提取指令字节
                if command == 0xA1:  #单目测距
                    current_task = 1
                elif command == 0xA2:  #标靶对正
                    current_task = 2
    except Exception as e:
        print(f"串口接收异常: {e}")
        current_task = 0

def put_chinese_text(img, text, position, font_path="simhei.ttf", font_size=30, color=(0, 0, 255)):
    # 将OpenCV图像转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    # 绘制文本
    draw.text(position, text, fill=color[::-1], font=font)  # PIL使用RGB，OpenCV使用BGR，需反转颜色
    # 转换回OpenCV格式
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv

def initialize_camera(camera_id, frame_size):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
    # 关闭自动曝光/自动白平衡（减少光线干扰）
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, 100)  # 固定曝光值，需根据环境微调
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)  # 固定白平衡
    return cap

# ====================== 主程序 ======================
if __name__ == "__main__":
    # 串口初始化
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        time.sleep(2)  # 串口启动延时
        print(f"成功连接串口 {SERIAL_PORT}")
    except Exception as e:
        print(f"串口连接异常: {e}")

    # 注册热键
    keyboard.add_hotkey('up', on_up)
    keyboard.add_hotkey('down', on_down)
    keyboard.add_hotkey('enter', on_enter)
    keyboard.add_hotkey('ctrl+c', on_ctrl_c)

    while True:
        # 检查是否按下了 ESC 键
        if keyboard.is_pressed('esc'):
            print("ESC key pressed, exiting...")
            break
        if not current_task:
            if ser and ser.is_open:
                receive_serial_command(ser)
        else:
            if current_task == 1:
                print("单目测距任务开始")
                print("========== 高精度测距操作指南 ==========")
                print("1. 标靶垂直地面放置，中心与摄像头大致同高")
                print("2. 测量范围：40-60cm，画面中绿色框为识别到的标靶")
                print("3. 按ctrl+c退出，按s保存标定帧，按c重新标定K值")
                print("=======================================\n")
                monocular_distance_measurement()
            elif current_task == 2:
                print("标靶对正任务开始")
                print("按q系统启动，开始追踪标靶")
                target_alignment()
    if ser and ser.is_open:
        ser.close()
        print("串口已关闭")
    print("系统正常退出")