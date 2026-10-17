import requests

def send_close_signal():
    """发送关门信号"""
    try:
        # 设置设备的IP地址和控制路径
        device_ip = '192.168.1.100'  # 修改为实际设备的IP地址
        url = f'http://{device_ip}/close'  # 修改为实际的控制URL
        
        # 发送关门指令
        response = requests.get(url)
        print("关门信号已发送")
        
    except Exception as e:
        print(f"发送关门信号时出错: {str(e)}")
