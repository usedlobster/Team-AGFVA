#!/usr/bin/env python
import numpy as np
import copy # TODO Needed?
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = .5

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        
        # Add a subscriber for /obstacle_waypoint below
        # rospy.Subscriber('/obstacle_waypoint', Int32, self.traffic_cb) TODO - implement

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Other member variables you need below
        self.base_waypoints = None
        self.pose = None
        self.stopline_wp_idx = -1
        self.start_slow_wp_idx = -1 # TODO needed??
        self.waypoints_2d = None
        self.waypoint_tree = None

        self.loop()
 
    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Check if closest waypoint is ahead of behind the vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coords
        closest_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(closest_vect - prev_vect, pos_vect - closest_vect)

        # If the closest waypoint is behind us...
        if val > 0:
            # ...take the next one
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        #final_lane.header.frame_id = '/world'        # TODO Needed?
        #final_lane.header.stamp = rospy.Time.now()   # TODO Needed?
        #final_lane.header.seq = self.msg_seq_num     # TODO Needed?
        #self.msg_seq_num += 1                        # TODO Needed?
        self.final_waypoints_pub.publish(final_lane)        
        
    def generate_lane(self):
      lane = Lane()

      closest_idx = self.get_closest_waypoint_idx()
      far_idx = closest_idx + LOOKAHEAD_WPS
      #rospy.loginfo_throttle( 1 , self.base_waypoints.waypoints[closest_idx].twist.twist )        
      next_waypoints = self.base_waypoints.waypoints[closest_idx:far_idx] 

      if self.stopline_wp_idx == -1 or ( self.stopline_wp_idx >= far_idx ):
          # rospy.loginfo_throttle(1,'normal!')
          lane.waypoints = next_waypoints 
      else:
          # rospy.loginfo_throttle(1,'slowing!')
          lane.waypoints = self.decelerate_waypoints( next_waypoints , closest_idx )
      return lane  
 
    def decelerate_waypoints( self , waypoints , closest_idx ):
        # TODO If we have a large list of waypoints we want to reduce/slice this
        # list so that the processing in this function does not introduce too much
        # latency     
        temp = [] 

        # Three waypoints back from line so front of car stops at line
        stop_idx =  max( self.stopline_wp_idx - closest_idx - 3 , 0 )  
    
        #rospy.loginfo( self.distance( waypoints , 0 , stop_idx ))
        for i , wp in enumerate( waypoints ) :
            # lesson to self , dont do p = wp !!!
            p = Waypoint()
            p.pose = wp.pose 

            dist = self.distance( waypoints , i , stop_idx )
            # vel = math.sqrt(2 * MAX_DECEL * dist)  # TODO adapt to linear or s-shaped curve to make breaking more comfortable
            vel = math.sqrt( 2 * dist )
            if ( vel < 1.0 ) :
                vel =0.0 
     
            # dont exceed original waypoint velocity 
            p.twist.twist.linear.x = max( 0 , min( vel , wp.twist.twist.linear.x ))            
            p.twist.twist.angular.z = 0.0   # TODO needed??

            temp.append( p ) 

        return temp 

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):  
        self.waypoints_2d = [[w.pose.pose.position.x, w.pose.pose.position.y] for w in waypoints.waypoints]
        self.waypoint_tree = KDTree(self.waypoints_2d)
        self.base_waypoints = waypoints 
        
    def traffic_cb(self, msg):
        if ( self.stopline_wp_idx != msg.data ) :
            # rospy.loginfo( 'stopline @'  + str(msg.data) )
            self.stopline_wp_idx = msg.data
            self.start_slow_wp_idx = -1   # TODO Needed?

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass 

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
