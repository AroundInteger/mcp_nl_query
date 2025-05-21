clear;clc;close ALL

% Set up the data
num_players = 11;
num_features = 20;  % 10 for home player, 10 for away player
num_matches = num_players * 2 - 2;  % Each player plays against every other player twice (home and away)
total_matches = num_players * num_matches;

% Generate more structured sample data
% We'll create 3 distinct player types
player_types = randi(3, num_players, 1);
X = zeros(total_matches, num_features);
match_results = zeros(total_matches, 1);  % Store match results
match_details = cell(total_matches, 1);  % Store detailed match information

for i = 1:num_players
    for j = 1:num_players
        if i == j
            continue;
        end
        
        % Home game
        home_match_index = (i-1)*num_matches + (j-1)*2 + 1;
        
        % Away game
        away_match_index = (j-1)*num_matches + (i-1)*2 + 2;
        
        % Generate base performance for both players
        home_performance = randn(1, 10) * 0.5;
        away_performance = randn(1, 10) * 0.5;
        
        % Add type-specific strengths
        if player_types(i) == 1
            % Offensive player
            home_performance(1:5) = home_performance(1:5) + 2;
        elseif player_types(i) == 2
            % Defensive player
            home_performance(6:10) = home_performance(6:10) + 2;
        else
            % All-round player
            home_performance = home_performance + 1;
        end
        
        if player_types(j) == 1
            % Offensive player
            away_performance(1:5) = away_performance(1:5) + 2;
        elseif player_types(j) == 2
            % Defensive player
            away_performance(6:10) = away_performance(6:10) + 2;
        else
            % All-round player
            away_performance = away_performance + 1;
        end
        
        % Add some randomness to represent match-day performance
        home_performance = home_performance + randn(1, 10) * 0.2;
        away_performance = away_performance + randn(1, 10) * 0.2;
        
        % Store the performances
        X(home_match_index, 1:10) = home_performance;
        X(home_match_index, 11:20) = away_performance;
        
        X(away_match_index, 1:10) = away_performance;
        X(away_match_index, 11:20) = home_performance;
        
        % Determine match result for home game
        [home_result, home_score, away_score] = determine_winner(home_performance, away_performance);
        match_results(home_match_index) = home_result;
        if home_result == 1
            result_str = 'HOME WIN';
        else
            result_str = 'AWAY WIN';
        end
        match_details{home_match_index} = sprintf('Player %d (HOME) vs Player %d: %s (Score: %.2f - %.2f)', ...
            i, j, result_str, home_score, away_score);
        
        % Determine match result for away game
        [away_result, away_score, home_score] = determine_winner(away_performance, home_performance);
        match_results(away_match_index) = away_result;
        if away_result == 1
            result_str = 'AWAY WIN';
        else
            result_str = 'HOME WIN';
        end
        match_details{away_match_index} = sprintf('Player %d vs Player %d (HOME): %s (Score: %.2f - %.2f)', ...
            i, j, result_str, away_score, home_score);
    end
end

% Calculate win percentages for each player
win_percentages = zeros(num_players, 1);
for i = 1:num_players
    player_matches = ((i-1)*num_matches + 1):((i-1)*num_matches + num_matches);
    wins = sum(match_results(player_matches) == 1);
    win_percentages(i) = wins / num_matches;
end

% Step 1: Data Preparation
X_centered = X - mean(X);

% Step 2: Perform SVD
[U, S, V] = svd(X_centered, 'econ');

% Step 3: Calculate explained variance ratio
singular_values = diag(S);
explained_variance = singular_values.^2 / (total_matches - 1);
total_variance = sum(explained_variance);
explained_variance_ratio = explained_variance / total_variance;
cumulative_variance_ratio = cumsum(explained_variance_ratio);

% Step 4: Select Principal Components
num_components = find(cumulative_variance_ratio >= 0.90, 1);

% Step 5: Project Data
X_pca = X_centered * V(:, 1:num_components);

% Step 6: Project singular values onto player features
player_styles = zeros(num_players, num_components);
for i = 1:num_players
    player_matches = ((i-1)*num_matches + 1):((i-1)*num_matches + num_matches);
    player_home_performances = X_pca(player_matches(1:2:end), :);
    player_away_performances = X_pca(player_matches(2:2:end), :);
    player_styles(i, :) = mean([player_home_performances; player_away_performances], 1);
end

% Step 7: Cluster player styles using DBSCAN
[idx, epsilon, minPts] = find_dbscan_params(player_styles, 3);

% Step 8: Interpret Results
% Plot explained variance ratio
figure;
bar(explained_variance_ratio);
title('Explained Variance Ratio');
xlabel('Principal Component');
ylabel('Explained Variance Ratio');

% Scree plot
figure;
plot(1:length(singular_values), singular_values, 'bo-');
title('Scree Plot');
xlabel('Principal Component');
ylabel('Singular Value');

% Heatmap of player styles
figure;
heatmap(player_styles);
title('Player Styles based on Principal Components');
xlabel('Principal Components');
ylabel('Players');
colorbar;

% Scatter plot of first two PCs for players, colored by cluster
figure;
gscatter(player_styles(:,1), player_styles(:,2), idx);
title('Player Styles Clustered (DBSCAN)');
xlabel('PC1');
ylabel('PC2');
legend('Location', 'bestoutside');

% Biplot of first two PCs for players
figure;
biplot(V(:,1:2), 'Scores', player_styles(:,1:2), 'VarLabels', cellstr(num2str((1:num_features)')));
title('Biplot of PC1 and PC2 for Player Styles');

% Print summary
fprintf('Number of components selected: %d\n', num_components);
fprintf('Cumulative explained variance: %.2f%%\n', 100*cumulative_variance_ratio(num_components));
fprintf('DBSCAN parameters: epsilon = %.2f, minPts = %d\n', epsilon, minPts);

% Display loadings
disp('Loadings (coefficients of original features in each PC):');
disp(V(:, 1:num_components));

% Analyze and print cluster characteristics
num_clusters = max(idx);
for i = 1:num_clusters
    cluster_members = find(idx == i);
    cluster_style = mean(player_styles(cluster_members, :), 1);
    cluster_win_percentage = mean(win_percentages(cluster_members));
    
    fprintf('\nCluster %d Characteristics:\n', i);
    [sorted_values, sorted_indices] = sort(abs(cluster_style), 'descend');
    for j = 1:min(5, num_components)
        pc_index = sorted_indices(j);
        fprintf('PC%d: %.2f\n', pc_index, cluster_style(pc_index));
        
        [~, top_features] = sort(abs(V(:, pc_index)), 'descend');
        fprintf('  Top features: %s\n', num2str(top_features(1:3)'));
    end
    
    fprintf('Players in this cluster: %s\n', num2str(cluster_members));
    fprintf('Cluster Win Percentage: %.2f%%\n', cluster_win_percentage * 100);
end

% Print match results
fprintf('\nMatch Results:\n');
for i = 1:total_matches
    fprintf('%s\n', match_details{i});
end

% Prepare data for Random Forest
X_rf = X; % Use all features
y_rf = match_results; % Use match results as the response variable

% Split data into training and testing sets
cv = cvpartition(size(X_rf, 1), 'HoldOut', 0.3);
idx_train = cv.training;
idx_test = cv.test;

X_train = X_rf(idx_train, :);
y_train = y_rf(idx_train);
X_test = X_rf(idx_test, :);
y_test = y_rf(idx_test);

% Train Random Forest model
num_trees = 100;
rf_model = TreeBagger(num_trees, X_train, y_train, 'Method', 'classification', 'OOBPredictorImportance', 'on');

% rf_model = fitcensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', num_trees, 'Learners', 'tree', 'Type', 'classification');
% feature_importance = oobPermutedPredictorImportance(rf_model);

% Make predictions on test set
[y_pred, scores] = predict(rf_model, X_test);
y_pred = str2double(y_pred);

% Calculate accuracy
accuracy = sum(y_pred == y_test) / length(y_test);
fprintf('\nRandom Forest Accuracy: %.2f%%\n', accuracy * 100);

%% Calculate feature importance
feature_importance = rf_model.OOBPermutedVarDeltaError;
%feature_importance = predictorImportance(rf_model);
%feature_importance =oobPermutedPredictorImportance(rf_model)
%%
% Visualize feature importance
figure;
bar(feature_importance);
title('Feature Importance');
xlabel('Feature Index');
ylabel('Importance');
xticks(1:num_features);
xticklabels(1:num_features);
xtickangle(45);

% Visualize predicted vs actual responses
figure;
scatter(y_test, y_pred, 'filled');
hold on;
plot([0 1], [0 1], 'r--');
xlabel('Actual Response');
ylabel('Predicted Response');
title('Predicted vs Actual Response');
legend('Predictions', 'Perfect Prediction', 'Location', 'southeast');

% Calculate confusion matrix
conf_matrix = confusionmat(y_test, y_pred);

% Visualize confusion matrix
figure;
confusionchart(conf_matrix);
title('Confusion Matrix');

% Calculate and print additional metrics
precision = conf_matrix(2,2) / sum(conf_matrix(:,2));
recall = conf_matrix(2,2) / sum(conf_matrix(2,:));
f1_score = 2 * (precision * recall) / (precision + recall);

fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1_score);

% % ROC curve
% [fpr, tpr, thresholds] = roc(y_test, scores(:,2));
% figure;
% plot(fpr, tpr);
% hold on;
% plot([0 1], [0 1], 'r--');
% xlabel('False Positive Rate');
% ylabel('True Positive Rate');
% title('ROC Curve');
% legend('ROC Curve', 'Random Guess', 'Location', 'southeast');
% 
% % Calculate AUC
% auc = trapz(fpr, tpr);
% fprintf('Area Under the Curve (AUC): %.2f\n', auc);
%% Calculate ROC curve
[X, Y, T, AUC] = perfcurve(y_test, scores(:,2), '1');

% Plot ROC curve
figure;
plot(X, Y);
hold on;
plot([0 1], [0 1], 'r--');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');
legend('ROC Curve', 'Random Guess', 'Location', 'southeast');

% Display AUC
text(0.6, 0.2, sprintf('AUC = %.3f', AUC), 'FontSize', 12);

% Print AUC
fprintf('Area Under the Curve (AUC): %.3f\n', AUC);

% % Save results including Random Forest analysis
% save('pca_results_svd_players_dbscan_rf.mat', 'X_pca', 'V', 'explained_variance_ratio', ...
%     'cumulative_variance_ratio', 'player_styles', 'idx', 'epsilon', 'minPts', ...
%     'win_percentages', 'match_results', 'match_details', 'rf_model', 'accuracy', ...
%     'feature_importance', 'conf_matrix', 'precision', 'recall', 'f1_score', 'auc');

% Function to determine the winner of a match
function [result, home_score, away_score] = determine_winner(home_performance, away_performance)
    home_score = sum(home_performance);
    away_score = sum(away_performance);
    
    % Add a small home advantage
    home_advantage = 0.5;
    
    if home_score + home_advantage > away_score
        result = 1;  % Home win
    else
        result = 0;  % Away win
    end
end

% Function to find appropriate DBSCAN parameters
function [idx, best_epsilon, best_minPts] = find_dbscan_params(data, target_clusters)
    best_diff = Inf;
    best_epsilon = 0;
    best_minPts = 0;
    best_idx = [];
    
    for epsilon = 0.1:0.1:2
        for minPts = 2:5
            idx = dbscan(data, epsilon, minPts);
            num_clusters = length(unique(idx(idx ~= -1)));
            if num_clusters > 0 && abs(num_clusters - target_clusters) < best_diff
                best_diff = abs(num_clusters - target_clusters);
                best_epsilon = epsilon;
                best_minPts = minPts;
                best_idx = idx;
                if num_clusters == target_clusters
                    return;
                end
            end
        end
    end
    idx = best_idx;
end