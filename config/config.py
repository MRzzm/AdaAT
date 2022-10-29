import argparse

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def add_base(self):
        self.parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
        self.parser.add_argument('--img_channel', type=int, default=3, help='channels of input image')



class PersonTrainingOptions(BaseOptions):
    def parse_args(self):
        self.add_base()
        self.parser.add_argument('--keypoint_num', type=int, default=18, help='num of 2d keypoints')
        self.parser.add_argument('--train_data', type=str, default=r"./assert/fasion_train_data.json",
                            help='json path of train data')
        self.parser.add_argument('--train_img_dir', type=str, default=r"",
                            help='img dir of train data')
        self.parser.add_argument('--start_epoch', default=1, type=int, help='start epoch in training stage')
        self.parser.add_argument('--non_decay', default=200, type=int, help='num of epoches with fixed learning rate')
        self.parser.add_argument('--decay', default=200, type=int, help='num of linearly decay epochs')
        self.parser.add_argument('--batch_size', type=int, default=24, help='training batch size')
        self.parser.add_argument('--checkpoint', type=int, default=2, help='num of checkpoints in training stage')
        self.parser.add_argument('--lamb_perception', type=int, default=5, help='weight of perception loss')
        self.parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate of generator')
        self.parser.add_argument('--lr_d', type=float, default=0.0001, help='learning rate of discriminator')
        self.parser.add_argument('--result_path', type=str, default=r"./assert/result/person_deepFashion",
                                 help='result path to save model')
        self.parser.add_argument('--resume', type=str, default='None',
                                 help='If true, load model and training.')
        self.parser.add_argument('--resume_path',
                                 default='',
                                 type=str,
                                 help='Save data (.pth) of previous training')
        # =========================  Discriminator ==========================
        self.parser.add_argument('--D_num_blocks', type=int, default=4, help='num of down blocks in discriminator')
        self.parser.add_argument('--D_block_expansion', type=int, default=64, help='block expansion in discriminator')
        self.parser.add_argument('--D_max_features', type=int, default=256, help='max channels in discriminator')
        return self.parser.parse_args()

class PersonInferenceOptions(BaseOptions):
    def parse_args(self):
        self.add_base()
        self.parser.add_argument('--keypoint_num', type=int, default=18, help='num of 2d keypoints')
        self.parser.add_argument('--inference_model_path', type=str, default="./assert/person_epoch_30.pth",
                            help='path of trained model')
        self.parser.add_argument('--source_img_path', type=str, default="./assert/example_person_source_img.jpg",
                            help='path of source image')
        self.parser.add_argument('--source_kp_path', type=str, default="./assert/example_person_souce_kp.txt",
                            help='path of source key points')
        self.parser.add_argument('--target_kp_path', type=str, default="./assert/example_person_target_kp.txt",
                            help='path of target key points')
        self.parser.add_argument('--res_person_path', type=str, default="./assert/example_person_inference_img.jpg",
                            help='path of output person image')
        return self.parser.parse_args()

class MetricOptions(BaseOptions):
    def parse_args(self):
        self.parser.add_argument('--inference_img_dir', type=str, default=r"",
                        help='dir path of inference images')
        self.parser.add_argument('--real_img_dir', type=str, default=r"",
                            help='dir path of real images')
        self.parser.add_argument('--task_type', type=str, default="",
                                 help='face or person')
        return self.parser.parse_args()

class FaceInference256Options(BaseOptions):
    def parse_args(self):
        self.add_base()
        self.parser.add_argument('--keypoint_num', type=int, default=68, help='num of 2d keypoints')
        self.parser.add_argument('--inference_model_path', type=str, default="./assert/face_256.pth",
                            help='path of trained model')
        self.parser.add_argument('--img_size', type=int, default=256, help='the size of img')
        self.parser.add_argument('--source_img_path', type=str, default="./assert/example_face_source_img.jpg",
                            help='path of source image')
        self.parser.add_argument('--control_part', type=str, default="expression;pose",#expression;pose
                                 help='controllable semantic facial parts, including expression and pose')
        self.parser.add_argument('--res_face_path', type=str,
                                 default="./assert/example_face_inference_video_reenactment.mp4",
                                 help='path of output face video')
        #####################################  face reenactment ######################################
        self.parser.add_argument('--driving_video_path', type=str, default="./assert/example_face_driving_video.mp4",
                                 help='path of driving video in face reenactment')
        self.parser.add_argument('--driving_kp_path', type=str, default="./assert/example_face_driving_video1.csv",
                            help='path of driving key points,using openface (option)')
        return self.parser.parse_args()


class FaceInference512Options(BaseOptions):
    def parse_args(self):
        self.add_base()
        self.parser.add_argument('--keypoint_num', type=int, default=68, help='num of 2d keypoints')
        self.parser.add_argument('--inference_model_path', type=str, default="./assert/face_512_epoch_66.pth",
                            help='path of trained model')
        self.parser.add_argument('--img_size', type=int, default=512, help='the size of img')
        self.parser.add_argument('--source_img_path', type=str, default="./assert/example_face_source_img.jpg",
                            help='path of source image')
        self.parser.add_argument('--driving_audio_path', type=str, default="./assert/audio/Obam_03.wav",
                                 help='path of driving audio in taling face generation')
        self.parser.add_argument('--mouth_model_path', type=str, default="./assert/audio2animation_mouth.pth",
                                 help='pretrained model of mouth animation generator')
        self.parser.add_argument('--eyebrow_model_path', type=str, default="./assert/audio2animation_eyebrow.pth",
                                 help='pretrained model of eyebrow animation generator')
        self.parser.add_argument('--pose_model_path', type=str, default="./assert/audio2animation_pose.pth",
                                 help='pretrained model of pose animation generator')
        self.parser.add_argument('--emotion', type=str, default="neutral",
                                 help='set emotion in animation generator,including angry,happy,neutral,surprise')
        self.parser.add_argument('--deep_speech_model_path', type=str, default="./assert/deep_speech.pb",
                                 help='pretrained model of deep speech feature extractor')
        self.parser.add_argument('--res_face_path', type=str, default="./assert/audio/Obam_03.mp4",
                                 help='path of output face video')
        return self.parser.parse_args()




