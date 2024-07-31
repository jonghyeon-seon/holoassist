# ICCV 2023 HoloAssist:  an egocentric human interaction dataset for interactive ai assistants in the real world 

The codebase provides guidelines for using the HoloAssist dataset and running the benchmarks. 

[[Project Website](https://holoassist.github.io/)][[paper](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_HoloAssist_an_Egocentric_Human_Interaction_Dataset_for_Interactive_AI_Assistants_ICCV_2023_paper.html)]


# Download the data and annotations 

We release the dataset under the [[CDLAv2](https://cdla.dev/permissive-2-0/)] license, a permissive license. 
<!-- You can download the data and annotations via the links in the text files below. You can either downloading the data through your web browser -->
<!-- or using [[Azcopy](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10)] which will be **faster**.  -->

<!-- 
**Install Azcopy and download data via Azcopy in Linux.**  

Please refer to the [official manual](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10) of using Azcopy in other OS. 
```bash
- wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
- tar -xvf azcopy.tar.gz
- sudo mv azcopy_linux_amd64_*/azcopy /usr/bin
- azcopy --version
```
Downloading the data 
```bash
- azcopy copy "<data_url>" "<local_directory>" --recursive
```
 -->

## Dataset Structure
Once the dataset is downloaded and decompressed. You will see the dataset structure as follows. Each folder contains data for one recording session. Within each folder, you will see the data for different modalities. The text files with "_synced" in the names are synced according to the RGB modality as each modality has different sensor rate and we use the synced modalities in the experiments.  

We collected our dataset using [PSI studio](https://github.com/microsoft/psi/tree/master/Sources/MixedReality/HoloLensCapture). More detailed information regarding the data format is in [here](https://github.com/microsoft/psi/tree/master/Sources/MixedReality/HoloLensCapture/HoloLensCaptureExporter).


<pre>
.
├── R007-7July-DSLR
│   └── Export_py
│       │── AhatDepth
│       │   ├── 000000.png
│       │   ├── 000001.png
│       │   ├── ...
│       │   ├── AhatDepth_synced.txt
│       │   ├── Instrinsics.txt
│       │   ├── Pose_sync.txt
│       │   └── Timing_sync.txt
│       ├── Eyes
│       │   └── Eyes_sync.txt 
│       ├── Hands
│       │   ├── Left_sync.txt
│       │   └── Right_sync.txt 
│       ├── Head
│       │   └── Head_sync.txt 
│       ├── IMU
│       │   ├── Accelerometer_sync.txt
│       │   ├── Gyroscope_sync.txt
│       │   └── Magnetometer_sync.txt
│       ├── Video
│       │   ├── Pose_sync.txt
│       │   ├── Instrinsincs.txt
│       │   └── VideoMp4Timing.txt
│       ├── Video_pitchshift.mp4
│       └── Video_compress.mp4
├── R012-7July-Nespresso/
├── R013-7July-Nespresso/
├── R014-7July-DSLR/
└── ...
</pre>



## Annotation Structure

We have released both the annotations in the raw format and the processed format. We also provide the train, validation and test splits. 

In the raw annotations, each annotation follows 
<pre>
{
    "id": int, original label id,
    "label": "Narration", "Conversation", "Fine grained action",  or "Coarse grained action", 
    "start": start time in seconds, 
    "end": end time in seconds, 
    "type":"range",
    "attributes":{
        Different from different label task. See below.
    },
},
</pre>

Attributes for **Narration**
<pre>
    "id": int, original label id,
    "label": "Narration",  
    "start": start time in seconds, 
    "end": end time in seconds, 
    "type":"range",
    "attributes": {
        "Long-form description": Use multiple sentences and make this as long as is necessary to be exhaustive. There are a finite number of scenarios across all videos, so make sure to call out the distinctive changes between videos, in particular, mistakes that the task performer makes in the learning process that are either self-corrected or corrected by the instructor.
    }, 
</pre>
- **Example**:
A man operates a big office printer. The instructor provides directions on how to turn on and load paper into the big office printer. The man turns on the printer and then turns it off. He then loads paper into the first drawer of the printer and replaces the first black cartridge from left to right in the printer. The instructor corrects the man on where to place the first black cartridge. The man moves the black cartridge from its current position to the correct location.
Note: The time stamps for this annotation will always start at 0 and end at the end of the video.

Attributes for **Conversation**
<pre>
    "id": int, original label id,
    "label": "Narration",  
    "start": start time in seconds, 
    "end": end time in seconds, 
    "type":"range",
    "attributes": {
        "Conversation Purpose":"instructor-start-conversation_other",
        "Transcription":"*unintelligible*",
        "Transcription Confidence":"low-confidence-transcription",
    }, 
</pre>

- **Conversation Purpose**: Select an option that best describes the purpose of the speech. This is limited to the individual speaking and does not include any pause time waiting for a response.
    - Instructor-start-conversation:  Describing high-level instruction 
    - Instructor-start-conversation:   Opening remarks
    - Instructor-start-conversation:   Closing remarks
    - Instructor-start-conversation:  Adjusting to capture better quality video
    - Instructor-start-conversation:  Confirming the previous or future action
    - Instructor-start-conversation:   Correct the wrong action
    - Instructor-start-conversation:   Follow-up instruction
    - Instructor-start-conversation: Other 
    - Instructor-reply-to-task performer: Confirming the previous or future action
    - Instructor-reply-to-task performer: Correct the wrong action
    - Instructor-reply-to-task performer: Follow-up instruction
    - Instructor-reply-to-task performer: other
    - task performer-start-conversation: ask questions
    - task performer-start-conversation: others

- **Transcriptions**: Transcribe the conversation into texts.
- **Transcription Confidence**: Confidence for the human annotator at translating the speech to text. 

Attributes for **Fine grained action**
<pre>
    "id": int, original label id,
    "label": "Fine grained action",  
    "start": start time in seconds, 
    "end": end time in seconds, 
    "type":"range",
    "attributes": {
        "Action Correctness":"Correct Action",
        "Incorrect Action Explanation":"none",
        "Incorrect Action Corrected by":"none",
        "Verb":"approach",
        "Adjective":"none",
        "Noun":"gopro",
        "adverbial":"none"
    }, 
</pre>

- **Action Correctness**: Indicate whether the action is correct or a mistake to achieve the task. The options are 
    - Correct action
    - Wrong action, corrected by instructor verbally
    - Wrong action, corrected by performer
    - Wrong action, not corrected
    - Others
    
- **Incorrect Action Explanation**: Provided by the human annotators to explain why the believe the action is wrong. 
- **Incorrect Action Correct by**: Indicate whether the wrong action is later corrected by the instructor or the task performer. 
- **Verb**, **Adjective**, **Noun**, **adverbial**: Verb, (adjective), Noun, (adverbial) for describing the fine-grained actions. 

Attributes for **Coarse grained action**
<pre>
    "id": int, original label id,
    "label": "Coarse grained action",  
    "start": start time in seconds, 
    "end": end time in seconds, 
    "type":"range",
    "attributes": {
        "Action sentence":"The student changes the battery for the GoPro.",
        "Verb":"exchange",
        "Adjective":"none",
        "Noun":"battery"
    }, 
</pre>

- **Action sentence**: A factual statement describing the interaction that the collector/camera wearer is performing with a digital device and the software on the device. 
    - Example 1: A man changes the battery of the bright green GoPro.
    - Example 2: A woman attaches the leg to a chair.
- **Verb**: This verb was part of the Coarse-Grained Action sentence.
    - Example 1: Change
    - Example 2: Attach
- **Adjective**: This is the adjective(s) that helps distinguish the noun from other similar items. This field is optional if the noun is unique enough on its own. 
    - Example 1: bright green
    - Example 2: [blank]
- **Noun**: This is the generic noun that is part of the Coarse-Grained Action sentence.
    - Example 1: GoPro 
    - Example 2: Leg



**Citation**


If you find the code or data useful. Please consider cite the paper at
```
@inproceedings{wang2023holoassist,
  title={Holoassist: an egocentric human interaction dataset for interactive ai assistants in the real world},
  author={Wang, Xin and Kwon, Taein and Rad, Mahdi and Pan, Bowen and Chakraborty, Ishani and Andrist, Sean and Bohus, Dan and Feniello, Ashley and Tekin, Bugra and Frujeri, Felipe Vieira and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20270--20281},
  year={2023}
}
```