<script>
    import { onMount } from 'svelte';
    import * as PIXI from 'pixi.js';
    let canvas;
    onMount(async () => {
        const app = new PIXI.Application();
        await app.init({ view: canvas, width: 640, height: 360, backgroundColor: 0x1099bb });
        await PIXI.Assets.load('/sample.png');
        let sprite = PIXI.Sprite.from('/sample.png');
        app.stage.addChild(sprite);
        app.ticker.add((ticker) => {
            sprite.rotation +=  0.01 * ticker.deltaTime;
        });
    });

</script>


<canvas class="h-64 w-64" id="app" bind:this={canvas}></canvas>


<h1>Welcome to SvelteKit</h1>
<p>Visit <a href="https://kit.svelte.dev">kit.svelte.dev</a> to read the documentation</p>
