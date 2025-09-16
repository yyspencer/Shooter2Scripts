#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <limits>
#include <iomanip>
#include <cmath>  // For variance calculation
#include <algorithm>  // For trimming whitespace
#include <boost/math/distributions/students_t.hpp> 
using namespace std;
namespace fs = filesystem;
using namespace boost::math;
//g++ shookpupil.cpp -I/opt/homebrew/include -L/opt/homebrew/lib -lboost_math_c99
string trim(const string& str) {
    string trimmed = str;
    trimmed.erase(trimmed.begin(), find_if(trimmed.begin(), trimmed.end(), [](unsigned char ch) {
        return !isspace(ch);
    }));
    trimmed.erase(find_if(trimmed.rbegin(), trimmed.rend(), [](unsigned char ch) {
        return !isspace(ch);
    }).base(), trimmed.end());
    return trimmed;
}

// Function to check if a file has a .csv extension
bool isCSVFile(const fs::path& filePath) {
    return filePath.extension() == ".csv";
}

// Function to load CSV file into a 2D vector
vector<vector<string>> loadCSV(const string& filePath) {
    vector<vector<string>> data;
    ifstream file(filePath);
    if (!file.is_open()) {
        cerr << "Error: Could not open " << filePath << endl;
        return {};
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<string> row;
        string cell;
        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        data.push_back(row);
    }

    file.close();
    return data;
}

// Function to find the column that contains "leftpupil" and "rightpupil"
pair<int, int> findPupilColumns(const vector<string>& headerRow) {
    int leftPupilCol = -1, rightPupilCol = -1;
    for (size_t i = 0; i < headerRow.size(); i++) {
        string trimmedCell = trim(headerRow[i]);
        if (trimmedCell.find("leftPupil") != string::npos) {
            leftPupilCol = i;
        }
        if (trimmedCell.find("rightPupil") != string::npos) {
            rightPupilCol = i;
        }
    }
    return {leftPupilCol, rightPupilCol};
}

// Function to find the row index of "0.2 seconds" and "shook"
pair<int, int> findEventRows(const vector<vector<string>>& data, int eventColumn) {
    int rowFor02 = -1, rowForShook = -1;
    for (size_t i = 1; i < data.size(); i++) { // Start from row 1 to skip header
        if (data[i].size() <= eventColumn) continue;
        string eventColumnValue = data[i][eventColumn];

        if (eventColumnValue.find("0.2 seconds") != string::npos && rowFor02 == -1) {
            rowFor02 = i;
        }
        if (eventColumnValue.find("shook") != string::npos && rowForShook == -1) {
            rowForShook = i;
        }
        if (rowFor02 != -1 && rowForShook != -1) break;
    }

    return {rowFor02, rowForShook};
}

// Function to compute pupil size averages before and after events
vector<double> calculatePupilAverages(const vector<vector<string>>& data, int timeCol, int leftPupilCol, int rightPupilCol, int rowFor02, int rowForShook) {
    double sumLeftBefore = 0.0, sumRightBefore = 0.0, leftcountBefore = 0, rightcountBefore = 0;
    double sumLeftAfter = 0.0, sumRightAfter = 0.0, leftcountAfter = 0, rightcountAfter = 0, beforecount=0, aftercount=0;

    double timeBefore = stod(data[rowFor02][timeCol]);
    double timeAfter = stod(data[rowForShook][timeCol]);

    for (size_t i = 1; i < data.size(); i++) { // Start from row 1 to skip header
        if (data[i].size() <= max(leftPupilCol, rightPupilCol)) continue;

        double timeValue, leftPupilSize, rightPupilSize;
        try {
            timeValue = stod(data[i][timeCol]);
            leftPupilSize = stod(data[i][leftPupilCol]);
            rightPupilSize = stod(data[i][rightPupilCol]);
        } catch (...) {
            continue;
        }

        if (timeValue >= (timeBefore - 5.0) && timeValue <= timeBefore) {
            if (leftPupilSize>0){
                sumLeftBefore += leftPupilSize;
                leftcountBefore++;
            }
            if (rightPupilSize>0){
                sumRightBefore += rightPupilSize;
                rightcountBefore++;
            }
        }
        
        if (timeValue >= timeAfter && timeValue <= (timeAfter + 5.0)) {
            if (leftPupilSize>0){
                sumLeftAfter += leftPupilSize;
                leftcountAfter++;
            }
            if (rightPupilSize>0){
                sumRightAfter += rightPupilSize;
                rightcountAfter++;
            }
        }
    }
    double avgLeftBefore = (leftcountBefore > 0) ? sumLeftBefore / leftcountBefore : -1;
    double avgRightBefore = (rightcountBefore > 0) ? sumRightBefore / rightcountBefore : -1;
    double avgLeftAfter = (leftcountAfter > 0) ? sumLeftAfter / leftcountAfter : -1;
    double avgRightAfter = (rightcountAfter > 0) ? sumRightAfter / rightcountAfter : -1;
    if ((leftcountBefore/beforecount)<0.5){
        avgLeftBefore=-1;
    }
    if ((rightcountBefore/beforecount)<0.5){
        avgRightBefore=-1;
    }
    if ((leftcountAfter/aftercount)<0.5){
        avgLeftAfter=-1;
    }
    if ((rightcountAfter/aftercount)<0.5){
        avgRightAfter=-1;
    }
    return {avgLeftBefore, avgRightBefore, avgLeftAfter, avgRightAfter};
}
struct PupilData {
        string index;
        bool isShook;
        double datalist[4];
};
int main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);

    string path = ".";
    fs::path shookFolder = fs::path(path) / "shook";

    cout << fixed << setprecision(3);
    cout << "Scanning CSV files in the shook folder..." << endl;

    if (!fs::exists(shookFolder) || !fs::is_directory(shookFolder)) {
        cerr << "Error: 'shook' folder does not exist!" << endl;
        return 1;
    }

    cout << "\n==== Pupil Analysis Report ====\n";
    int totalcnt=0;
    for (const auto& entry : fs::directory_iterator(shookFolder)) {
        if (fs::is_regular_file(entry.path()) && isCSVFile(entry.path())) {
            totalcnt++;
        }
    }
    vector<PupilData> database;
    vector<pair<double, double>> leftpupil, rightpupil;
    for (const auto& entry : fs::directory_iterator(shookFolder)) {
        if (fs::is_regular_file(entry.path()) && isCSVFile(entry.path())) {
            string fileName = entry.path().filename().string();
            string fileIndex = fileName.substr(0, 5);

            // Load CSV file into 2D vector
            vector<vector<string>> data = loadCSV(entry.path().string());
            if (data.empty()) {
                cout << "Index " << fileIndex << " -> ERROR: Could not load CSV ❌" << endl;
                continue;
            }

            // Find column indices
            pair<int, int> pupilColumns = findPupilColumns(data[0]);
            if (pupilColumns.first == -1 || pupilColumns.second == -1) {
                cout << "Index " << fileIndex << " -> ERROR: 'leftpupil' or 'rightpupil' column not found ❌" << endl;
                continue;
            }
            int eventColumn = -1;
            for (size_t i = 0; i < data[0].size(); i++) {
                if (trim(data[0][i]).find("robotEvent") != string::npos) {
                    eventColumn = i;
                    break;
                }
            }
            if (eventColumn == -1) {
                cout << "Index " << fileIndex << " -> ERROR: 'robotEvent' column not found ❌" << endl;
                continue;
            }

            // Find row indices
            pair<int, int> eventRows = findEventRows(data, eventColumn);
            if (eventRows.first == -1 || eventRows.second == -1) {
                cout << "Index " << fileIndex << " -> ERROR: '0.2 seconds' or 'shook' not found ❌" << endl;
                continue;
            }

            // Calculate pupil size averages
            vector<double> pupilAverages = calculatePupilAverages(data, 0, pupilColumns.first, pupilColumns.second, eventRows.first, eventRows.second);
            PupilData newdata;
            newdata.index=fileIndex;
            newdata.isShook=true;
            newdata.datalist[0] = pupilAverages[0];
            newdata.datalist[1] = pupilAverages[1];
            newdata.datalist[2] = pupilAverages[2];
            newdata.datalist[3] = pupilAverages[3];
            database.push_back(newdata);
            // Check for negative averages
            cout<<"Index "<<fileIndex<<" -> ";
            if (pupilAverages[0]<0||pupilAverages[2]<0){
                cout<<"Invalid left eye ❌, ";
            }
            else{
                cout<<"Valid left eye ✅ ";
                leftpupil.push_back(make_pair(pupilAverages[0], pupilAverages[2]));
            }
            if (pupilAverages[1]<0||pupilAverages[3]<0){
                cout<<"Invalid right eye ❌, "<<'\n';
            }
            else{
                cout<<"Valid right eye ✅ "<<'\n';
                rightpupil.push_back(make_pair(pupilAverages[1], pupilAverages[3]));
            }
        }
    }
    double lvar = 0, rvar = 0, leftbefore = 0, leftafter = 0, rightbefore = 0, rightafter = 0;

    // Compute mean for left pupil
    for (const auto& i : leftpupil) {
        leftbefore += i.first;
        leftafter += i.second;
    }
    leftbefore /= leftpupil.size();
    leftafter /= leftpupil.size();

    // Compute mean for right pupil
    for (const auto& i : rightpupil) {
        rightbefore += i.first;
        rightafter += i.second;
    }
    rightbefore /= rightpupil.size();
    rightafter /= rightpupil.size();

    // Compute variance for left pupil
    for (const auto& i : leftpupil) {
        lvar += pow(((i.second - i.first) - (leftafter - leftbefore)), 2);
    }
    lvar/=rightpupil.size()-1;
    // Compute variance for right pupil
    for (const auto& i : rightpupil) {
        rvar += pow(((i.second - i.first) - (rightafter - rightbefore)), 2);
    }
    rvar/=rightpupil.size()-1;
    // Standard deviation
    double lsd = sqrt(lvar);
    double rsd = sqrt(rvar);

    // Compute t-scores
    double ltscore = (leftafter-leftbefore)/(lsd/sqrt(leftpupil.size()));
    double rtscore = (leftafter-leftbefore)/(rsd/sqrt(rightpupil.size()));

    // Display statistics
    cout <<"T-scores: Left = " << ltscore << ", Right = " << rtscore << '\n';
    cout << "Valid left count: " << leftpupil.size()<<"/"<<totalcnt<<" "<< ", Valid right count: " << rightpupil.size()<< "/"<<totalcnt<<'\n';
    cout << "Avg Left Before: " << leftbefore << ", Avg Left After: " << leftafter 
         << ", Avg Left Diff: " << (leftafter - leftbefore) << '\n';
    cout << "Avg Right Before: " << rightbefore << ", Avg Right After: " << rightafter 
         << ", Avg Right Diff: " << (rightafter - rightbefore) << '\n';

    // Perform t-test
    double alpha;
    cout << "Enter significance level: ";
    cin >> alpha;

    students_t ldist(leftpupil.size() - 1);
    students_t rdist(rightpupil.size() - 1);
    double lp = 1 - cdf(ldist, ltscore);
    double rp = 1 - cdf(rdist, rtscore);
    cout<<"LP: "<<lp<<" RP: "<<rp<<'\n';
    if (lp < alpha) {
        cout << "Left Reject the null hypothesis (Significant result) ✅" << endl;
    } else {
        cout << "Left Fail to reject the null hypothesis (Not significant) ❌" << endl;
    }
    if (rp < alpha) {
        cout << "Right Reject the null hypothesis (Significant result) ✅" << endl;
    } else {
        cout << "Right Fail to reject the null hypothesis (Not significant) ❌" << endl;
    }
    return 0;
}