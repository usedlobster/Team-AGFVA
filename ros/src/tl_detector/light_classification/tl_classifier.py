import tensorflow as tf
import cv2
import numpy as np
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        self.threshold = .01
        self.img = None

        PATH_TO_MODEL = '../../../training_folder/fine_tuned_model_sim_5000/frozen_inference_graph.pb'
 		
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
	# tensor flow can get greedy with gpu memory 
	# limit to 60% 
	# should be done as rosparam  
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.6


        self.sess = tf.Session(config=config,graph=self.detection_graph)

        # TODO: This is just a guess. Values have to be reviewed after training
        self.tl_mapping = {
            1 : TrafficLight.GREEN,
            2 : TrafficLight.RED,
            3 : TrafficLight.YELLOW
        }

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        tl_state = TrafficLight.UNKNOWN

        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(image, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
        
        # return boxes, scores, classes, num
        if num == 0:
            # No bounding boxes found
            tl_state = TrafficLight.UNKNOWN
        
        max_score = scores[0][0] 
        prediction = classes[0][0] 
        print("max_score: {}, prediction: {}".format(max_score, prediction))

        ## ---------------------------------------------------------------------
        # Code for debugging TODO remove or comment later
        for box, score, clas in zip(boxes[0], scores[0], classes[0]):
            if score > self.threshold:
                print "Score: ", score, "Class: ", clas, "Box: ", ["{0:.2f}".format(b) for b in box]
                # cv2.rectangle(image, (box[0],box[1]),(box[2],box[3]),(255,0,0),3)
        
        # cv2.rectangle(image, (4,4),(200,200),(255,0,0),3)
        (rows,cols,channels) = image.shape
        print rows, cols, channels
        # cv2.imshow("name", cv_image)
        # cv2.waitKey(3)
        cv2.imwrite("test.jpg", image)

        # if self.img is None:
        #     self.img = pl.imshow(image)
        # else:
        #     self.img.set_data(image)
        #     pl.pause(.1)
        #     pl.draw()

        ## ---------------------------------------------------------------------
        



        if max_score > self.threshold and prediction >= 0 and prediction <= 2:
            tl_state = self.tl_mapping[int(prediction)]
        
        print("STATE: %s", tl_state)
        return tl_state 
