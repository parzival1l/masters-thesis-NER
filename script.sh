#!/bin/bash

LOG_FILE="launch_experiment.log"
SPECIFIC_WORD="python"
SPECIFIC_SCRIPT_EXT=".py"
DEBUG_MODE=false
IS_LOCAL=false
USER_NAME=''
EXPERIMENT_NAME=''
COMMAND_LIST=()

set -e # To stop the main if any function hit exit

display_help() {
  bold=$(tput bold)
  normal=$(tput sgr0)
  underline=$(tput smul)
  reset_underline=$(tput rmul)
  italic_start="\033[3m"
  italic_end="\033[0m"

  echo " ${bold}Presentation: ${normal}" 
  echo " This script allows executing various experiments in an Azure Databricks compute cluster." 
  echo " It was developed to address the following encountered issues:"
  echo " - To be able to launch multiple experiments using hydra-core, we need to initiate the experimentation via the terminal."
  echo " - Code execution via the terminal is limited to 12 hours."
  echo " - The use of tmux does not allow code execution in subdirectories: ${bold}Workspace/REPOS/..${normal}"  
  echo ""
  echo " ${bold}What this script does:${normal}"
  echo " This script launches experiments and maintains their execution until they are completed to overcome the 12-hour limit."
  echo " To launch the experimentation, the script copies your code to the DBFS data server and starts the execution."
  echo " When the experiment is completed, the folder created in DBFS is ${bold}not automatically deleted.${normal}"
  echo " Finally, a file named ${bold}launch_experiment.log${normal} will be created in your ${bold}Workspace/REPOS/..${normal} directory to indicate whether the execution of your experiment was successful or not."
  echo ""
  echo "${bold}How to execute this script:${normal}"
  echo "1. Open the cluster's web terminal."
  echo "2. Set the directory to the actual repo path: "
  echo -e "   ${italic_start}cd /Workspace/Repos/user@7dish.com/xxx/src/Spikes/xxx ${italic_end}"
  echo "3. Run the command:" 
  echo -e "   ${italic_start}databricks configure --token ${italic_end}"
  echo "4. Then execute the script:" 
  echo -e "   ${italic_start}./script.sh -n simon -e ingredient ''python src/download_data.py'' ${italic_end}"
  echo ""
  echo "${bold}Usage:${normal}"
  echo "  ./script.sh [options] [command(s)]"
  echo ""
  echo "${bold}Example of execution:${normal}"
  echo '  ./script.sh -n simon -e ingredient "python src/download_data.py" "python src/run_experiment.py param1=1,2,3 param2=1,2,3 -m"'
  echo ""
  echo "${bold}Options:${normal}"
  echo "  ${underline}-n | --name <name> (Mandatory)${reset_underline}"
  echo "    Specify a user name to create a working directory."
  echo "    Working directory structure: <name>/<experiment>/<time-stamp>"
  echo ""
  echo "  ${underline}-e | --exp <experiment> (Mandatory)${reset_underline}"
  echo "    Specify an experiment name to create a working directory."
  echo "    Working directory structure: <name>/<experiment>/<time-stamp>"
  echo ""
  echo "  ${underline}-d | --debug <debug> (Optional)${reset_underline}"
  echo "    Enable debugging mode (toggle switch)."
  echo "    When enabled, the script attaches to a tmux panel and keeps it open."
  echo "    Useful for viewing terminal outputs; disables automatic tmux closure."
  echo ""
  echo "  ${underline}-l | --local <local> (Optional)${reset_underline}"
  echo "    Execute the script locally (toggle switch)."
  echo "    When enabled, the working directory is created in the parent directory."
  echo "    Useful for testing the script in a local environment, rather than on az-databricks."
  echo ""
  echo "  ${underline}-h | --help <help>${reset_underline}"
  echo "    Display this help message."
  echo ""
  echo "${bold}Commands:${normal}"
  echo ""
  echo "  ${underline}A command${reset_underline}"
  echo "    The command to be passed is the hydra-core command that initiates an experiment."
  echo '    Command example: "python src/main.py param1=1,2,3 param2=1,2,3 -m"'
  echo ""
  echo ""
  exit 0
}

log_message() {
  local message="$1"
  printf "$(date +"%Y-%m-%d %H:%M:%S") - $message \n" >> "$LOG_FILE"
}

function create_working_directory_in_dbfs() {
    mkdir -p "$1"
}

function move_files_to_dbfs() {
    zip_file="experiment.zip"
    zip -r "$zip_file" . -x ./notebooks/* > /dev/null 2>&1 # Zip everything in the current directory
    cp "$zip_file" "$1" # Copy the zip file to the destination
    unzip -d "$1" "$1/$zip_file" > /dev/null 2>&1 # Unzip the zip file in the destination directory
    rm "$zip_file"
}

function evaluate_params_and_option() {
  # Process options using getopts
  args=$(getopt -a -o hdln:e: --long help,debug,local,name:,exp: -- "$@")
  if [[ $? -gt 0 ]]; then
    display_help
  fi
  eval set -- ${args}
  user_name_provided=false
  experiment_name_provided=false

  while :
  do
  log_message "$1"
    case $1 in
      -h | --help)
        log_message "$1" ;
        display_help ;
        exit 0
        ;;
      -d | --debug)
        DEBUG_MODE=true;
        shift;
        ;;
      -l | --local)
        IS_LOCAL=true;
        shift;
        ;;
      -n | --name)
        USER_NAME=$2;
        user_name_provided=true;
        shift 2;
        ;;
      -e | --exp)
        EXPERIMENT_NAME=$2;
        experiment_name_provided=true;
        shift 2;
        ;;
      \?)
        log_message "Invalid option: -$OPTARG";
        exit 1;
        ;;
        # -- means the end of the arguments; drop this, and break out of the while loop
        --) shift; break ;;
    esac
  done

  # Print debug mode status
  if ${DEBUG_MODE}; then
    log_message "Debug mode is enabled."
  else
    log_message "Debug mode is disabled."
  fi

  # Print local mode status
  if ${IS_LOCAL}; then
    log_message "Is local mode is enabled."
  else
    log_message "Is local mode is disabled."
  fi

  # Check if mandatory options are provided
  if [ "$user_name_provided" = false ] || [ "$experiment_name_provided" = false ]; then
    echo "Both -n and -e options are mandatory."
    echo "Usage: $0 [-n myName] [-e myExperimentName]"
    exit 1
  fi
  # Shift to the next argument after processing options
  shift $((OPTIND-1))

  # Loop through remaining parameters
  if [[ "$#" -eq 0 ]]; then
        log_message "Error: At least one parameter is required."
        exit 1
  fi

  
  for param in "$@"; do
    # Check if the parameter contains the specific word
    if ! [[ "${param}" == "${SPECIFIC_WORD}"*"${SPECIFIC_SCRIPT_EXT}"* ]]; then
            log_message "ERROR: The parameter should be a python script execution.
            The parameter should contains '${SPECIFIC_WORD}' and '${SPECIFIC_SCRIPT_EXT}'.
            Parameter sent: '${param}'"
            exit 1
    else
        COMMAND_LIST+=("${param}")
    fi
  done
}

function launch_tmux_session_og() {
    for (( i = 0; i < ${#COMMAND_LIST[@]}; i++ )); do
        SECONDS=0
        tmux_session="experiment_$((i+1))"
        log_message "Tmux session name $tmux_session run command: '${COMMAND_LIST[i]}'"
        #tmux new-session -c $1 -s $tmux_session -d "${COMMAND_LIST[i]};  tmux wait -L experiment"
        tmux new-session -c $1 -d -s $tmux_session # ajout
        tmux set-option -g mouse on

        # tmux wait -L experiment # ajout
        tmux send-keys -t "$tmux_session" C-z "${COMMAND_LIST[i]}" Enter # ajout
        tmux attach-session -t $tmux_session

        # tmux wait experiment
        tmux send-keys -t "$tmux_session" exit

        duration=$SECONDS
        log_message "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
        # Check the exit status of the tmux command
        if ! [ $? -eq 0 ]; then
            log_message "ERROR: Tmux session name $tmux_session run command: '${COMMAND_LIST[i]}', failded"
        fi
        
    done
}

function launch_tmux_session() {
    for (( i = 0; i < ${#COMMAND_LIST[@]}; i++ )); do
        SECONDS=0
        tmux_session="experiment_$((i+1))"
        log_message "Tmux session name $tmux_session run command: '${COMMAND_LIST[i]}'"

        tmux new-session -c $1 -d -s $tmux_session # ajout
        tmux set-option -g mouse on
        
        if ${DEBUG_MODE}; then
            tmux send-keys -t "$tmux_session" C-z "${COMMAND_LIST[i]}" Enter # ajout
            tmux attach-session -t $tmux_session  # fait en sorte d<ouvrir le terminal
        else 
            tmux wait -L experiment
            tmux send-keys -t "$tmux_session" C-z "${COMMAND_LIST[i]}" Enter # ajout
            tmux send-keys -t "$tmux_session" "exit" Enter
            tmux wait experiment
        fi

        duration=$SECONDS
        log_message "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
        # Check the exit status of the tmux command
        if ! [ $? -eq 0 ]; then
            log_message "ERROR: Tmux session name $tmux_session run command: '${COMMAND_LIST[i]}', failded"
        fi
        
    done
}

function main() {
    log_message "Step 1: Check option and params..."
    evaluate_params_and_option "$@"
    
    _base_dir_path="${USER_NAME}/${EXPERIMENT_NAME}"
    if ${IS_LOCAL}; then
        timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
        tmp_directory="../${_base_dir_path}/${timestamp}"
    else
        # Create the tmp_directory with the timestamp
        timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
        tmp_directory="/dbfs/${_base_dir_path}/${timestamp}"
        #tmp_directory="${folder_name}"
    fi
    log_message "Step 2: Create the working dirctory..."
    log_message "Temporary folder created at: ${tmp_directory}"
    create_working_directory_in_dbfs "${tmp_directory}"
    
    log_message "Step 3: Move files to dbfs..."
    move_files_to_dbfs "${tmp_directory}"

    log_message "Step 4: Launch tmux session..."
    launch_tmux_session "${tmp_directory}"

    log_message "Step 5: Remove temporary diretory..."
    trap "exit 1"           HUP INT PIPE QUIT TERM

    # trap 'rm -rf "${tmp_directory}"'  EXIT
    log_message "Step 5: Completed"
}

# Invoke main with args if not sourced
# Approach via: https://stackoverflow.com/a/28776166/8787985
if ! (return 0 2> /dev/null); then
    main "$@"
fi