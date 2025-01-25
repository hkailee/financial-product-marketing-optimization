# Pipelines

## Global Pipeline Templates

To use the global pipeline templates, you simply have to add the repo as a resource and reference the template like this:

```yaml
## we need to reference the Template Repo here
resources:
  repositories:
    - repository: pipeline-templates
      type: git
      name: pipeline-templates
      ref: refs/tags/v1.0 # Tagged Version of Templates
# ...
- template: pipelines/template_build.yml@pipeline-templates
  parameters: 
            # ...
```
