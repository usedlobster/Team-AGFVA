import tensorflow as tf
import numpy as np
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        self.threshold = .01

        PATH_TO_MODEL = '../../../training_folder/fine_tuned_model_sim_200/frozen_inference_graph.pb'
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
        self.sess = tf.Session(graph=self.detection_graph)

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
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(image, axis=0)  
            (_, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
        
        if num == 0:
            # No bounding boxes found
            return TrafficLight.UNKNOWN
        
        max_score = scores[0][0]
        prediction = classes[0][0]
        print("max_score: {}, prediction: {}".format(max_score, prediction))
        
        if max_score > self.threshold and prediction >= 0 and prediction <= 2:
            return self.tl_mapping[prediction]
        
        return TrafficLight.UNKNOWN
