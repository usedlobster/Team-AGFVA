import time, rospy
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter


GAS_DENSITY = 2.858
ONE_MPH     = 0.44704
MAX_SPEED   = 18.2 # kph set in waypoint loader


class Controller(object):
    def __init__(self, vehicle_mass, brake_deadband, wheel_radius, decel_limit, wheel_base, 
        steer_ratio, max_lat_accel, max_steer_angle):

        min_speed           = 1.0 * ONE_MPH

        kp         = 0.3
        ki         = 0.1
        kd         = 0.
	mn 	   = 0.
        mx         = 0.2
        self.throttle_pid   = PID(kp, ki, kd, mn, mx)
        tau = 0.5
        ts  = .02
        self.vel_lpf = LowPassFilter(tau, ts)
        self.yaw_control    = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        self.brake_deadband = brake_deadband
        self.vehicle_mass   = vehicle_mass
        self.wheel_radius   = wheel_radius
        self.decel_limit    = decel_limit

        self.last_time      = None

    def control(self, target_v, target_omega, current_v, dbw_enabled):

        # Calculate the desired throttle based on target_v, target_omega and current_v
        # target_v and target_omega are desired linear and angular velocities

        if self.last_time is None or not dbw_enabled:
            self.last_time = time.time()
            return 0.0, 0.0, 0.0

	current_v.x = self.vel_lpf.filt(current_v.x)
        steer = self.yaw_control.get_steering(target_v.x, target_omega.z, current_v.x)

        dt = time.time() - self.last_time

        # Assumed maximum speed is in mph

        error = min(target_v.x, MAX_SPEED) - current_v.x

        throttle = self.throttle_pid.step(error, dt)
	brake = 0;
	
	if target_v.x == 0. and current_v.x < 0.1:
		throttle = 0
		brake = 400

	elif throttle < .1 and error < 0:
		throttle = 0
		decel = max(error, self.decel_limit)
		brake = abs(decel) * self.wheel_radius * self.vehicle_mass 
 
        self.last_time = time.time()

        return throttle, brake, steer