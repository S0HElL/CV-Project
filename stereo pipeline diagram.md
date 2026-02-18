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
        LoadCalib[Load Calibration]:::process
        CalcParams[Get Parameters<br/>focal_length f, baseline B]:::process
    end

    Start --> LoadConfig --> LoadCalib --> CalcParams

    %% --- FRAME LOOP ---
    subgraph FrameLoop [Per-Frame Processing]
        
        LoadImages[Load Stereo Pair<br/>Left Img, Right Img]:::process
        CalcParams --> LoadImages
        
        %% --- STEREO MATCHING BRANCH ---
        subgraph Matching [Step 1 - Stereo Matching]
            MethodCheck{Method<br/>Configured?}:::decision
            
            %% Path A: SGBM
            subgraph SGBM_Path [Method - SGBM]
                ComputeSGBM[compute_disparity_sgbm]:::process
                SGBMNote[Internal Consistency Check]:::data
            end
            
            %% Path B: Block Matching
            subgraph BM_Path [Method - Block Matching]
                ComputeBM_LR[Compute L to R<br/>SAD, SSD, or NCC]:::process
                ComputeBM_RL[Compute R to L<br/>SAD, SSD, or NCC]:::process
                ConsistencyCheck[LR Consistency Check<br/>relax_threshold]:::process
                
                ComputeBM_LR & ComputeBM_RL --> ConsistencyCheck
            end
            
            MethodCheck -- SGBM --> ComputeSGBM
            ComputeSGBM --- SGBMNote
            MethodCheck -- block_matching --> ComputeBM_LR
        end
        
        LoadImages --> MethodCheck
        
        %% Convergence
        RawDisp(Raw or Consistent Disparity):::data
        SGBMNote --> RawDisp
        ConsistencyCheck --> RawDisp

        %% --- POST PROCESSING ---
        subgraph PostProcessing [Step 3 - Post-Processing]
            PP_Func[postprocess_disparity]:::process
            
            Filter1[Left-Right Invalid Masking]
            Filter2[Median Filtering]
            Filter3[Remove Speckles]
            Filter4[Hole Filling]
            
            PP_Func -.-> Filter1 & Filter2 & Filter3 & Filter4
        end
        
        RawDisp --> PP_Func
        ProcessedDisp(Processed Disparity Map):::data
        PP_Func --> ProcessedDisp

        %% --- DEPTH CONVERSION ---
        subgraph DepthConv [Step 2 - Depth Conversion]
            Eq[Apply Formula<br/>Z = f * B / d]
            CalcDepth[disparity_to_depth]:::process
        end
        
        ProcessedDisp --> Eq --> CalcDepth
        CalcParams -.-> Eq
        
        DepthMap(Depth Map):::data
        CalcDepth --> DepthMap

        %% --- OUTPUT ---
        subgraph Outputs [Saving and Visualization]
            SaveNPY[Save .npy Files]:::process
            VisPlots[Generate Plots<br/>plot_disparity and plot_depth]:::process
        end

        ProcessedDisp --> SaveNPY
        DepthMap --> SaveNPY
        ProcessedDisp --> VisPlots
        DepthMap --> VisPlots
    end

    VisPlots --> NextFrame{Next Frame?}:::decision
    NextFrame -- Yes --> LoadImages
    NextFrame -- No --> End([End]):::output
    
```