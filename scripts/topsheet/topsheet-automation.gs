/*
 * TopSheet Automation Script
 * 
 * Purpose: Google Apps Script (.gs) for automating TopSheet updates
 * Status: Implemented and working in Google Apps Script
 * Note: This file serves as historical record/proof of work
 * Usage: Copy this code to Google Apps Script editor to use
 * 
 * Features:
 * - Syncs TT-Forge project issues to Google Sheets
 * - Creates Key Milestones sheet automatically
 * - Filters issues by "Top Problems/Issues" label or Top Sheet field
 * - Custom menu integration in Google Sheets
 * 
 * Author: Sapna Khatri
 * Date: 08/29/2025
 */

const GITHUB_TOKEN = ""; // Set this in Google Apps Script Properties Service

// Add custom menu when spreadsheet opens
function onOpen() {
  const ui = SpreadsheetApp.getUi();
  ui.createMenu('TT-Forge Sync')
    .addItem('Update Topsheet', 'main')
    .addToUi();
}

function main() {
  try {
    Logger.log('🚀 Starting TT-Forge project sync...');
    
    // Get all qualifying issues from TT-Forge project
    const issues = getAllQualifyingIssues();
    
    Logger.log(`📊 Found ${issues.length} qualifying issues`);
    
    if (issues.length === 0) {
      Logger.log('⚠️  No qualifying issues found');
      return;
    }
    
    // Create or get sheet and write all data
    writeAllIssuesToSheet(issues);
    
    // Create Key Milestones sheet
    Logger.log('📋 Creating Key Milestones sheet...');
    createKeyMilestonesSheet();
    
    Logger.log(`✅ Successfully updated sheet with ${issues.length} issues from TT-Forge project`);
    Logger.log(`✅ Successfully created Key Milestones sheet`);
    
    // Show success message to user
    SpreadsheetApp.getUi().alert('Success!', 
      `Successfully updated sheet with ${issues.length} issues from TT-Forge project and created Key Milestones sheet.`, 
      SpreadsheetApp.getUi().ButtonSet.OK);
      
  } catch (error) {
    Logger.log('❌ Error: ' + error.toString());
    
    // Show error message to user
    SpreadsheetApp.getUi().alert('Error!', 
      `An error occurred: ${error.toString()}`, 
      SpreadsheetApp.getUi().ButtonSet.OK);
  }
}

function main() {
  try {
    Logger.log('🚀 Starting TT-Forge project sync...');
    
    // Get all qualifying issues from TT-Forge project
    const issues = getAllQualifyingIssues();
    
    Logger.log(`📊 Found ${issues.length} qualifying issues`);
    
    if (issues.length === 0) {
      Logger.log('⚠️  No qualifying issues found');
      return;
    }
    
    // Create or get sheet and write all data
    writeAllIssuesToSheet(issues);
    
    // Create Key Milestones sheet
    Logger.log('📋 Creating Key Milestones sheet...');
    createKeyMilestonesSheet();
    
    Logger.log(`✅ Successfully updated sheet with ${issues.length} issues from TT-Forge project`);
    Logger.log(`✅ Successfully created Key Milestones sheet`);
  } catch (error) {
    Logger.log('❌ Error: ' + error.toString());
  }
}

function getAllQualifyingIssues() {
  Logger.log('🔍 Searching TT-Forge project for qualifying issues...');
  
  const projectId = 'PVT_kwDOA9MHEM4AjeTl'; // TT-Forge project ID
  const TOP_SHEET_ID = 'PVTSSF_lADOA9MHEM4AjeTlzgsed74'; // Top Sheet field ID
  
  let qualifyingIssues = [];
  let hasNextPage = true;
  let endCursor = null;
  let totalSearched = 0;
  let pageCount = 0;
  
  while (hasNextPage && pageCount < 50) { // Limit to 50 pages (5000 items) for safety
    pageCount++;
    const afterClause = endCursor ? `, after: "${endCursor}"` : '';
    
    const query = `
      query {
        node(id: "${projectId}") {
          ... on ProjectV2 {
            items(first: 100${afterClause}) {
              pageInfo {
                hasNextPage
                endCursor
              }
              nodes {
                content {
                  ... on Issue {
                    number
                    title
                    url
                    state
                    createdAt
                    labels(first: 10) {
                      nodes {
                        name
                      }
                    }
                    assignees(first: 5) {
                      nodes {
                        login
                        name
                      }
                    }
                    repository {
                      name
                      owner {
                        login
                      }
                    }
                  }
                }
                fieldValues(first: 20) {
                  nodes {
                    ... on ProjectV2ItemFieldDateValue {
                      date
                      field {
                        ... on ProjectV2FieldCommon {
                          id
                          name
                        }
                      }
                    }
                    ... on ProjectV2ItemFieldTextValue {
                      text
                      field {
                        ... on ProjectV2FieldCommon {
                          id
                          name
                        }
                      }
                    }
                    ... on ProjectV2ItemFieldSingleSelectValue {
                      name
                      field {
                        ... on ProjectV2FieldCommon {
                          id
                          name
                        }
                      }
                    }
                    ... on ProjectV2ItemFieldNumberValue {
                      number
                      field {
                        ... on ProjectV2FieldCommon {
                          id
                          name
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    `;

    const options = {
      'method': 'POST',
      'headers': {
        'Authorization': `Bearer ${GITHUB_TOKEN}`,
        'Accept': 'application/vnd.github+json',
        'Content-Type': 'application/json'
      },
      'payload': JSON.stringify({ query: query })
    };

    Logger.log(`📄 Searching page ${pageCount}...`);
    const response = UrlFetchApp.fetch('https://api.github.com/graphql', options);
    const data = JSON.parse(response.getContentText());

    if (data.errors) {
      Logger.log('❌ GraphQL errors on page ' + pageCount + ':');
      for (let error of data.errors) {
        Logger.log(`  ${error.message}`);
      }
      break;
    }

    if (data.data && data.data.node && data.data.node.items) {
      const items = data.data.node.items.nodes || [];
      totalSearched += items.length;
      
      Logger.log(`📊 Page ${pageCount}: Found ${items.length} items (total searched: ${totalSearched})`);
      
      // Check each item for qualifying criteria
      for (let item of items) {
        if (item && item.content && item.content.number) {
          const issue = item.content;
          
          // Check criteria 1: Has "Top Problems/Issues" label
          const hasTopProblemsLabel = issue.labels && issue.labels.nodes && 
            issue.labels.nodes.some(label => label.name === 'Top Problems/Issues');
          
          // Check criteria 2: Has non-empty "Top Sheet" field
          let hasTopSheetValue = false;
          let customFields = {
            estimatedCompletion: '',
            statusUpdate: '',
            topSheet: ''
          };
          
          if (item.fieldValues && item.fieldValues.nodes) {
            for (let fieldValue of item.fieldValues.nodes) {
              if (fieldValue && fieldValue.field) {
                const fieldId = fieldValue.field.id;
                const value = fieldValue.date || fieldValue.text || fieldValue.name || fieldValue.number || '';
                
                if (fieldId === 'PVTF_lADOA9MHEM4AjeTlzgr-wr0') {
                  customFields.estimatedCompletion = value;
                } else if (fieldId === 'PVTF_lADOA9MHEM4AjeTlzgsD52g') {
                  customFields.statusUpdate = value;
                } else if (fieldId === TOP_SHEET_ID) {
                  customFields.topSheet = value;
                  if (value && value.trim() !== '') {
                    hasTopSheetValue = true;
                  }
                }
              }
            }
          }
          
          // If issue qualifies by either criteria
          if (hasTopProblemsLabel || hasTopSheetValue) {
            const qualifyingReason = [];
            if (hasTopProblemsLabel) qualifyingReason.push('has "Top Problems/Issues" label');
            if (hasTopSheetValue) qualifyingReason.push(`has Top Sheet value: "${customFields.topSheet}"`);
            
            Logger.log(`✅ Issue ${issue.number} qualifies: ${qualifyingReason.join(' and ')}`);
            
            qualifyingIssues.push({
              number: issue.number,
              title: issue.title,
              url: issue.url,
              state: issue.state,
              assignedTo: issue.assignees && issue.assignees.nodes ? 
                issue.assignees.nodes.map(a => a.name || a.login).join(', ') || 'Unassigned' : 'Unassigned',
              createdAt: new Date(issue.createdAt),
              labels: issue.labels && issue.labels.nodes ? 
                issue.labels.nodes.map(l => l.name).join(', ') : '',
              repository: issue.repository ? `${issue.repository.owner.login}/${issue.repository.name}` : 'unknown',
              customFields: customFields,
              qualifyingReason: qualifyingReason.join(' and ')
            });
          }
        }
      }
      
      // Update pagination info
      hasNextPage = data.data.node.items.pageInfo.hasNextPage;
      endCursor = data.data.node.items.pageInfo.endCursor;
      
    } else {
      Logger.log('❌ No project data returned on page ' + pageCount);
      break;
    }
  }
  
  Logger.log(`🎯 Found ${qualifyingIssues.length} qualifying issues out of ${totalSearched} total items`);
  return qualifyingIssues;
}

function writeAllIssuesToSheet(issues) {
  Logger.log(`📝 Writing ${issues.length} issues to Google Sheet...`);
  
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  
  // Get or create the TT-Forge Top Issues sheet
  let sheet = ss.getSheetByName('TT-Forge Top Issues');
  if (sheet) {
    // Sheet exists, clear it and reuse
    Logger.log('📋 TT-Forge Top Issues sheet exists, clearing and reusing...');
  } else {
    // Sheet doesn't exist, create it
    Logger.log('📋 Creating new TT-Forge Top Issues sheet...');
    sheet = ss.insertSheet('TT-Forge Top Issues');
  }
  
  // Clear any existing data
  sheet.clear();
  
  // Set up headers
  const headers = [
    'Issue Number',
    'Issue Title', 
    'Repository',
    'State',
    'Assigned To', 
    'Created At', 
    'Labels',
    'Estimated Completion Date', 
    'Status Update', 
    'Top Sheet',
    'Qualifying Reason',
    'Issue URL'
  ];
  
  const headerRange = sheet.getRange(1, 1, 1, headers.length);
  headerRange.setValues([headers]);
  headerRange.setFontWeight('bold');
  headerRange.setBackground('#4285f4');
  headerRange.setFontColor('white');
  
  // Set column widths
  sheet.setColumnWidth(1, 100);  // Issue Number
  sheet.setColumnWidth(2, 350);  // Issue Title
  sheet.setColumnWidth(3, 150);  // Repository
  sheet.setColumnWidth(4, 80);   // State
  sheet.setColumnWidth(5, 150);  // Assigned To
  sheet.setColumnWidth(6, 120);  // Created At
  sheet.setColumnWidth(7, 200);  // Labels
  sheet.setColumnWidth(8, 150);  // Estimated Completion Date
  sheet.setColumnWidth(9, 150);  // Status Update
  sheet.setColumnWidth(10, 150); // Top Sheet
  sheet.setColumnWidth(11, 250); // Qualifying Reason
  sheet.setColumnWidth(12, 300); // Issue URL
  
  // Freeze header row
  sheet.setFrozenRows(1);
  
  if (issues.length === 0) {
    Logger.log('⚠️  No issues to write');
    return;
  }
  
  // Prepare data rows
  const dataRows = issues.map(issue => [
    issue.number,
    issue.title,
    issue.repository,
    issue.state,
    issue.assignedTo,
    issue.createdAt,
    issue.labels,
    issue.customFields.estimatedCompletion,
    issue.customFields.statusUpdate,
    issue.customFields.topSheet,
    issue.qualifyingReason,
    issue.url
  ]);
  
  // Write all data at once for better performance
  const dataRange = sheet.getRange(2, 1, dataRows.length, headers.length);
  dataRange.setValues(dataRows);
  
  // Format dates (Created At column - column 6)
  if (dataRows.length > 0) {
    sheet.getRange(2, 6, dataRows.length, 1).setNumberFormat('mm/dd/yyyy');
    
    // Format Estimated Completion Date (column 8) if it has values
    sheet.getRange(2, 8, dataRows.length, 1).setNumberFormat('mm/dd/yyyy');
  }
  
  // Issue URLs are already in the data as raw URLs - no need for hyperlink formulas
  
  // Color code by state
  for (let i = 0; i < dataRows.length; i++) {
    const state = dataRows[i][3]; // State is at index 3
    const rowRange = sheet.getRange(2 + i, 1, 1, headers.length);
    
    if (state === 'CLOSED') {
      rowRange.setBackground('#f0f8f0'); // Light green for closed
    } else if (state === 'OPEN') {
      rowRange.setBackground('#fff8f0'); // Light orange for open
    }
  }
  
  // Auto-resize columns to fit content
  sheet.autoResizeColumns(1, headers.length);
  
  Logger.log(`✅ Successfully wrote ${issues.length} issues to sheet`);
}

function createKeyMilestonesSheet() {
  Logger.log('📋 Creating Key Milestones sheet...');
  
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  
  // Get data from TT-Forge Top Issues sheet
  const sourceSheet = ss.getSheetByName('TT-Forge Top Issues');
  if (!sourceSheet) {
    Logger.log('❌ TT-Forge Top Issues sheet not found');
    return;
  }
  
  // Get or create Key Milestones sheet
  let milestonesSheet = ss.getSheetByName('Key Milestones');
  if (milestonesSheet) {
    milestonesSheet.clear();
  } else {
    milestonesSheet = ss.insertSheet('Key Milestones');
  }
  
  // Get all data from source sheet (skip header row)
  const lastRow = sourceSheet.getLastRow();
  if (lastRow <= 1) {
    Logger.log('⚠️  No data found in TT-Forge Top Issues sheet');
    return;
  }
  
  const data = sourceSheet.getRange(2, 1, lastRow - 1, 12).getValues();
  
  // Group data by Top Sheet field (column J, index 9)
  const groupedData = {};
  
  data.forEach(row => {
    // Map columns: [Issue Number, Issue Title, Repository, State, Assigned To, Created At, Labels, Estimated Completion Date, Status Update, Top Sheet, Qualifying Reason, Issue URL]
    const [issueNumber, title, repository, state, assignedTo, createdAt, labels, estimatedCompletion, statusUpdate, topSheet, qualifyingReason, issueUrl] = row;
    
    if (!topSheet || topSheet.toString().trim() === '') {
      return; // Skip if no Top Sheet value
    }
    
    const topSheetKey = topSheet.toString().trim();
    
    if (!groupedData[topSheetKey]) {
      groupedData[topSheetKey] = {
        milestones: [],
        items: []
      };
    }
    
    // Check if this issue has "Top Problems/Issues" label
    const labelsStr = labels ? labels.toString() : '';
    const hasTopProblemsLabel = labelsStr.includes('Top Problems/Issues');
    
    if (hasTopProblemsLabel) {
      // Add to Top Problems/Issues section only
      groupedData[topSheetKey].items.push({
        item: title || '',
        owner: assignedTo || 'Unassigned',
        openDate: createdAt || '',
        dueDate: estimatedCompletion || '',
        status: statusUpdate || '',
        comment: '', // Empty for manual input
        url: issueUrl || '' // Store URL for hyperlink
      });
    } else {
      // Add to Key Upcoming Milestones/Deliverables section only
      groupedData[topSheetKey].milestones.push({
        when: estimatedCompletion || '',
        what: title || '',
        trend: statusUpdate || '',
        url: issueUrl || '' // Store URL for hyperlink
      });
    }
  });
  
  // Set up column widths
  milestonesSheet.setColumnWidth(1, 120);  // When
  milestonesSheet.setColumnWidth(2, 350);  // What
  milestonesSheet.setColumnWidth(3, 150);  // Trend
  milestonesSheet.setColumnWidth(4, 30);   // Gap
  milestonesSheet.setColumnWidth(5, 350);  // Items
  milestonesSheet.setColumnWidth(6, 120);  // Owner
  milestonesSheet.setColumnWidth(7, 100);  // Open Date
  milestonesSheet.setColumnWidth(8, 100);  // Due Date
  milestonesSheet.setColumnWidth(9, 150);  // Status
  milestonesSheet.setColumnWidth(10, 250); // Status Comment
  
  let currentRow = 1;
  
  // Create tables for each Top Sheet value
  for (const [topSheetValue, data] of Object.entries(groupedData)) {
    const milestones = data.milestones;
    const items = data.items;
    
    if (milestones.length === 0 && items.length === 0) continue;
    
    const sectionStartRow = currentRow;
    
    // Main header: "TopSheet Top Sheet Updated"
    const mainHeaderText = `Forge ${topSheetValue} Top Sheet Updated`;
    milestonesSheet.getRange(currentRow, 1, 1, 10).merge()
      .setValue(mainHeaderText)
      .setFontWeight('bold')
      .setFontSize(14)
      .setBackground('#93c47d')
      .setHorizontalAlignment('center')
      .setVerticalAlignment('middle');
    currentRow++;
    
    // Sub-headers row
    // Left: "Key Upcoming Milestones/Deliverables"
    milestonesSheet.getRange(currentRow, 1, 1, 3).merge()
      .setValue('Key Upcoming Milestones/Deliverables')
      .setFontWeight('bold')
      .setFontSize(11)
      .setBackground('#d9d9d9')
      .setHorizontalAlignment('center')
      .setVerticalAlignment('middle');
    
    // Right: "Top Problems/Issues"
    milestonesSheet.getRange(currentRow, 5, 1, 6).merge()
      .setValue('Top Problems/Issues')
      .setFontWeight('bold')
      .setFontSize(11)
      .setBackground('#d9d9d9')
      .setHorizontalAlignment('center')
      .setVerticalAlignment('middle');
    currentRow++;
    
    // Column headers
    // Left table headers
    const milestoneHeaders = ['When', 'What', 'Trend'];
    milestonesSheet.getRange(currentRow, 1, 1, 3).setValues([milestoneHeaders])
      .setFontWeight('bold')
      .setHorizontalAlignment('center')
      .setFontSize(10)
      .setBackground('#e6e6e6');
    
    // Right table headers
    const itemHeaders = ['Items', 'Owner', 'Open Date', 'Due Date', 'Status', 'Status Comment/Help Needed'];
    milestonesSheet.getRange(currentRow, 5, 1, 6).setValues([itemHeaders])
      .setFontWeight('bold')
      .setHorizontalAlignment('center')
      .setFontSize(10)
      .setBackground('#e6e6e6');
    currentRow++;
    
    // Determine max rows needed
    const maxRows = Math.max(milestones.length, items.length);
    
    // Add milestone data
    if (milestones.length > 0) {
      const milestoneData = milestones.map(m => [m.when, m.what, m.trend]);
      milestonesSheet.getRange(currentRow, 1, milestones.length, 3).setValues(milestoneData);
      
      // Add hyperlinks to What column (column 2)
      milestones.forEach((milestone, index) => {
        const cell = milestonesSheet.getRange(currentRow + index, 2);
        const title = milestone.what;
        const url = milestone.url;
        
        if (url && url.trim() !== '') {
          // Create RichTextValue with mixed formatting: "title tracker here" where only "tracker here" is hyperlinked
          const richText = SpreadsheetApp.newRichTextValue()
            .setText(`${title} tracker here`)
            .setLinkUrl(title.length + 1, title.length + 13, url) // " tracker here" positions
            .build();
          cell.setRichTextValue(richText);
        } else {
          // Fallback: just set the text without hyperlink
          cell.setValue(`${title} (no link)`);
        }
      });
      
      // Format milestone data
      const milestoneRange = milestonesSheet.getRange(currentRow, 1, milestones.length, 3);
      milestoneRange.setFontSize(10).setVerticalAlignment('top');
      
      // Format When column (dates)
      milestonesSheet.getRange(currentRow, 1, milestones.length, 1)
        .setNumberFormat('m/d')
        .setHorizontalAlignment('center');
      
      // Set text wrapping for What and Trend columns
      milestonesSheet.getRange(currentRow, 2, milestones.length, 1).setWrap(true);
      milestonesSheet.getRange(currentRow, 3, milestones.length, 1).setWrap(true);
      
      // Color code Trend column based on [G], [Y], [R], [D] prefixes
      const trendRange = milestonesSheet.getRange(currentRow, 3, milestones.length, 1);
      milestones.forEach((milestone, index) => {
        const trend = milestone.trend.toString();
        let bgColor = '#ffffff';
        if (trend.startsWith('[G]')) bgColor = '#93c47d';
        else if (trend.startsWith('[Y]')) bgColor = '#ffff00';
        else if (trend.startsWith('[R]')) bgColor = '#ff0000';
        else if (trend.startsWith('[D]')) bgColor = '#4185f4';
        
        milestonesSheet.getRange(currentRow + index, 3).setBackground(bgColor);
      });
    }
    
    // Add items data
    if (items.length > 0) {
      const itemData = items.map(item => [item.item, item.owner, item.openDate, item.dueDate, item.status, item.comment]);
      milestonesSheet.getRange(currentRow, 5, items.length, 6).setValues(itemData);
      
      // Add hyperlinks to Items column (column 5)
      items.forEach((item, index) => {
        const cell = milestonesSheet.getRange(currentRow + index, 5);
        const title = item.item;
        const url = item.url;
        
        if (url && url.trim() !== '') {
          // Create RichTextValue with mixed formatting: "title tracker here" where only "tracker here" is hyperlinked
          const richText = SpreadsheetApp.newRichTextValue()
            .setText(`${title} tracker here`)
            .setLinkUrl(title.length + 1, title.length + 13, url) // " tracker here" positions
            .build();
          cell.setRichTextValue(richText);
        } else {
          // Fallback: just set the text without hyperlink
          cell.setValue(`${title} (no link)`);
        }
      });
      
      // Format items data
      const itemRange = milestonesSheet.getRange(currentRow, 5, items.length, 6);
      itemRange.setFontSize(10).setVerticalAlignment('top');
      
      // Set text wrapping for relevant columns
      milestonesSheet.getRange(currentRow, 5, items.length, 1).setWrap(true);  // Items
      milestonesSheet.getRange(currentRow, 6, items.length, 1).setWrap(true);  // Owner
      milestonesSheet.getRange(currentRow, 9, items.length, 1).setWrap(true);  // Status
      milestonesSheet.getRange(currentRow, 10, items.length, 1).setWrap(true); // Comment
      
      // Format date columns
      milestonesSheet.getRange(currentRow, 7, items.length, 2)
        .setNumberFormat('m/d')
        .setHorizontalAlignment('center');
      
      // Color code Status column based on [G], [Y], [R], [D] prefixes
      items.forEach((item, index) => {
        const status = item.status.toString();
        let bgColor = '#ffffff';
        if (status.startsWith('[G]')) bgColor = '#93c47d';
        else if (status.startsWith('[Y]')) bgColor = '#ffff00';
        else if (status.startsWith('[R]')) bgColor = '#ff0000';
        else if (status.startsWith('[D]')) bgColor = '#4185f4';
        
        milestonesSheet.getRange(currentRow + index, 9).setBackground(bgColor);
      });
    }
    
    currentRow += maxRows + 2; // Add space between sections
  }
  
  Logger.log(`✅ Created Key Milestones sheet with ${Object.keys(groupedData).length} sections`);
}

// Standalone function to create only the Key Milestones sheet
function createMilestones() {
  try {
    Logger.log('📋 Creating Key Milestones sheet from existing data...');
    createKeyMilestonesSheet();
    Logger.log('✅ Key Milestones sheet created successfully');
  } catch (error) {
    Logger.log('❌ Error creating Key Milestones sheet: ' + error.toString());
  }
}