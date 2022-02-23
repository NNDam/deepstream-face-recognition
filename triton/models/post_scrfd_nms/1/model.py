import cv2
import numpy as np
from skimage import transform as trans
import triton_python_backend_utils as pb_utils

def align_face(img, bbox=None, landmark=None, image_size = (112, 112)):
  M = None
  if landmark is not None:
    assert len(image_size)==2
    src = np.array([
      [30.2946*image_size[0]/112, 51.6963*image_size[1]/112],
      [65.5318*image_size[0]/112, 51.5014*image_size[1]/112],
      [48.0252*image_size[0]/112, 71.7366*image_size[1]/112],
      [33.5493*image_size[0]/112, 92.3655*image_size[1]/112],
      [62.7299*image_size[0]/112, 92.2041*image_size[1]/112] ], dtype=np.float32 )

    src[:,0] += 8.0*image_size[1]/112
    dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
  if M is None:
    if bbox is None: #use center crop
      det = np.zeros(4, dtype=np.int32)
      det[0] = int(img.shape[1]*0.0625)
      det[1] = int(img.shape[0]*0.0625)
      det[2] = img.shape[1] - det[0]
      det[3] = img.shape[0] - det[1]
    else:
      det = bbox
    margin = kwargs.get('margin', 44*image_size[0]/112)
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
    bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
    ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
    if len(image_size)>0:
      ret = cv2.resize(ret, (image_size[1], image_size[0]))
    return ret 
  else: #do align using landmark
    assert len(image_size)==2
    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
    return warped

def perform_align(orig, num_detections, nmsed_boxes, nmsed_scores, nmsed_landmarks, input_size, threshold):
    """
        Return face aligned with bboxes, scores & landmarks
    """
    if num_detections == 0:
        return np.zeros([1, 4], dtype = np.float32), np.zeros([1, ], dtype = np.float32), np.zeros([1, 5, 2], dtype = np.float32), np.zeros((1, 3, 112, 112), dtype = np.float32)
    # Re-scale
    height, width, _ = orig.shape
    nmsed_boxes[:, 0]     *= width/input_size[0]
    nmsed_boxes[:, 1]     *= height/input_size[1]
    nmsed_boxes[:, 2]     *= width/input_size[0]
    nmsed_boxes[:, 3]     *= height/input_size[1]
    nmsed_landmarks           = nmsed_landmarks.reshape((-1, 5, 2))
    nmsed_landmarks[:, :, 0] *= width/input_size[0]
    nmsed_landmarks[:, :, 1] *= height/input_size[1]


    # Main process
    res_bboxes     = []
    res_scores     = []
    res_landmarks  = []
    res_face_align = []
    for i in range(num_detections):
        if nmsed_scores[i] < threshold:
            break
        res_bboxes.append(nmsed_boxes[i])
        res_scores.append(nmsed_scores[i])
        res_landmarks.append(nmsed_landmarks[i])
        res_face_align.append(align_face(orig, bbox = nmsed_boxes[i], landmark = nmsed_landmarks[i]))


    # Post-process
    if len(res_face_align) > 0:
        res_face_align = np.array(res_face_align, dtype = np.float32)
        res_face_align = res_face_align.transpose((0, 3, 1, 2))
        res_face_align = (res_face_align/255.0 - 0.5)/0.5
        res_bboxes     = np.array(res_bboxes, dtype = np.float32)
        res_scores    = np.array(res_scores, dtype = np.float32)
        res_landmarks = np.array(res_landmarks, dtype = np.float32)
        return res_bboxes, res_scores, res_landmarks, res_face_align
    else:
        return np.zeros([1, 4], dtype = np.float32), np.zeros([1, ], dtype = np.float32), np.zeros([1, 5, 2], dtype = np.float32), np.zeros((1, 3, 112, 112), dtype = np.float32)

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.input_size = (640, 640)
        self.threshold  = 0.6
        print('Initialized...')

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
        for request in requests:
            # Perform inference on the request and append it to responses list...
            # Get output tensors
            num_detections  = pb_utils.get_input_tensor_by_name(request, "num_detections").as_numpy()[0][0]  # [1, 1]
            nmsed_boxes     = pb_utils.get_input_tensor_by_name(request, "nmsed_boxes").as_numpy()[0]     # [1, 200, 4]
            nmsed_scores    = pb_utils.get_input_tensor_by_name(request, "nmsed_scores").as_numpy()[0]    # [1, 200,]
            nmsed_landmarks = pb_utils.get_input_tensor_by_name(request, "nmsed_landmarks").as_numpy()[0] # [1, 200, 10]
            original_image    = pb_utils.get_input_tensor_by_name(request, "original_image").as_numpy()[0] # [1, Width, height, 3]
            # print(original_image.shape, num_detections, nmsed_boxes)
            original_image = original_image.transpose((1, 2, 0))
            print('num_detections ', num_detections)
            res_bboxes, res_scores, res_landmarks, res_face_align = perform_align(orig = original_image,
                                                                                num_detections = num_detections,
                                                                                nmsed_boxes = nmsed_boxes,
                                                                                nmsed_scores = nmsed_scores,
                                                                                nmsed_landmarks = nmsed_landmarks,
                                                                                input_size = self.input_size,
                                                                                threshold=self.threshold)
            print(res_face_align.shape, res_bboxes.shape, res_scores.shape, res_landmarks.shape)
            res_num_detections     = pb_utils.Tensor("res_num_detections", np.array([len(res_bboxes)], dtype = np.int32))
            res_bboxes     = pb_utils.Tensor("res_bboxes", res_bboxes)
            res_scores     = pb_utils.Tensor("res_scores", res_scores)
            res_landmarks  = pb_utils.Tensor("res_landmarks", res_landmarks)
            res_face_align = pb_utils.Tensor("res_face_align", res_face_align)

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[res_num_detections, res_bboxes, res_scores, res_landmarks, res_face_align])
            responses.append(inference_response)

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')