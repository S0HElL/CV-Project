```Mermaid
flowchart TD
    %% --- STYLING DEFINITIONS ---
    classDef process fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:black
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:black
    classDef data fill:#e0f2f1,stroke:#00695c,stroke-width:2px,stroke-dasharray: 5 5,color:black
    classDef output fill:#f3e5f5,stroke:#880e4f,stroke-width:2px,color:black

    %% --- INITIALIZATION ---
    subgraph Init [Initialization]
        Start([Start]):::output
        LoadConfig[Load Config YAML]:::process
        LoadCalib[Load Calibration<br/>Get K, f, Baseline]:::process
        InitTraj[Initialize TrajectoryBuilder]:::process
    end

    Start --> LoadConfig --> LoadCalib --> InitTraj

    %% --- FRAME LOOP ---
    subgraph Loop [Per-Frame Processing Loop]
        
        LoadData[Load Image Data<br/>Left i, Right i, Left i+1]:::process
        InitTraj --> LoadData

        %% --- STEP 1: STEREO DEPTH ---
        subgraph StereoDepth [Step 1 - Stereo Depth for Scale]
            ComputeDisp[compute_disparity_optimized<br/>Block Matching]:::process
            DispMap(Disparity Map Frame i):::data
            
            ComputeDisp --> DispMap
        end

        %% --- STEP 2: FEATURES ---
        subgraph Features [Step 2 - Feature Matching]
            DetectMatch[detect_and_match<br/>Detect Features L_i and L_i+1<br/>Match Descriptors]:::process
            MatchData(Keypoints and Matches):::data
            
            DetectMatch --> MatchData
        end

        LoadData --> ComputeDisp
        LoadData --> DetectMatch

        %% --- STEP 3 & 4: POSE ESTIMATION ---
        subgraph PoseEst [Step 3 and 4 - Robust Pose Estimation]
            EstPose[estimate_pose_with_scale<br/>1. Back-project 2D i to 3D using Disparity<br/>2. Solve PnP RANSAC i to i+1]:::process
            
            DispMap --> EstPose
            MatchData --> EstPose
            LoadCalib -.-> EstPose
        end

        %% --- TRAJECTORY UPDATE ---
        CheckPose{Pose Found?}:::decision
        EstPose --> CheckPose
        
        AddPose[Update Trajectory<br/>trajectory.add_pose]:::process
        SkipPose[Skip / Add Identity]:::process
        
        CheckPose -- Yes --> AddPose
        CheckPose -- No --> SkipPose

        %% --- VISUALIZATION (Intermediate) ---
        VisMatches[Visualize Matches<br/>Save .png if i less than 3 or mod 10]:::process
        AddPose --> VisMatches
    end

    VisMatches --> NextFrame{Next Frame?}:::decision
    SkipPose --> NextFrame

    NextFrame -- Yes --> LoadData

    %% --- FINAL OUTPUTS ---
    subgraph Outputs [Step 5 - Trajectory Chaining and Output]
        SaveTxt[Save Trajectory .txt]:::process
        Plot2D[Plot Trajectory 2D<br/>Top View]:::process
        Plot3D[Plot Trajectory 3D]:::process
        End([End]):::output
    end

    NextFrame -- No --> SaveTxt
    SaveTxt --> Plot2D --> Plot3D --> End
```