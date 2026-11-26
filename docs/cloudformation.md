1. chainlit, when trying to add AWS Cognito as the OAuth provider, causes issues since the state path parameter in the callback is not compliant with AWS's requirements for base-64 encoded values. Thus, it needs a mockeypatch to force compliance and allow functioning with AWS Cognito.
2. You need to create the AWS DynamoDB table before use. The DynamoDB integration of Chainlit for persistence layer does not include table creation. Chainlit itself provides a json-based cloudformation template that can be used to create the table that chainlit requires.
3. When using the deploy command for cloudformation templates, the deployment is fully reversible (barring `DeletionPolicy: Retain` being present in the template):
    1. Deploy the template with the below command (with whatever stack name and template file and table name you want - can skip the parameters if there are none):
        `aws cloudformation deploy --stack-name my-dynamodb-stack --template-file your-template.json --parameter-overrides TableName=your-table-name`
    2. Revert the deployment with the below command (with same stack name which you used for deployment):
        `aws cloudformation delete-stack --stack-name my-dynamodb-stack`
4. Cloudformation templates (as long as no changes in the template) are also idempotent on using the deploy command.
5. In order to validate the cloudformation template syntax, use the below command (with your own path and extension - json or yaml):
    `aws cloudformation validate-template --template-body file://path-to-template.ext`
    And the output to the above command should be a list of parameters if you have any.
6. You can also use `cfn-lint` to do deeper linting (rather than just syntax).
7. You need to setup chainlit by first running `chainlit create-secret` and then writing the secret output to the .env file.
8. If you do the delete for the cloudformation template - then the dynamo db table is deleted without you paying anything
9. If you want to deploy multiple tables in multiple templates - best to use nested templates. deploy the templates in an s3 bucket and put the templates into a single template
