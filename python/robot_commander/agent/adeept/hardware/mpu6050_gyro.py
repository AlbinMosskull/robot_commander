import math

from mpu6050 import mpu6050


class Mpu6050Gyro:
    def __init__(self, i2c_address: int = 0x68):
        self._sensor = mpu6050(i2c_address)

    def z_angular_velocity_rad_s(self) -> float:
        gyro_data = self._sensor.get_gyro_data()
        return math.radians(gyro_data["z"])
