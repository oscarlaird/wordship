<script>
    import { onMount } from 'svelte';
    import { tweened } from 'svelte/motion';
    import { cubicOut } from 'svelte/easing';
    import * as PIXI from 'pixi.js';
    import Comm from './Comm.svelte';
    let canvas;
    let bottle_sprite;
    let ship_sprite;
    let ship_pos = {x: 1024.0 / 2.0, y: 1024.0 / 3.0};
    let tweened_ship_pos = tweened(ship_pos, {duration: 1000, easing: cubicOut});
    $: if (ship_sprite) ship_sprite.position = $tweened_ship_pos;

    let score = 0;

    let topics = [
        [{left: "evil", right: "good", top: "tame", bottom: "chaotic", category: "politics"}, {left: "evil", right: "good", top: "tame", bottom: "chaotic", category: "politics"}],
        [{left: "evil", right: "good", top: "tame", bottom: "chaotic", category: "politics"}, {left: "evil", right: "good", top: "tame", bottom: "chaotic", category: "politics"}]
    ]
    let words_to_vecs;

    // let dummy_island_names_list = [dummy_island_names_1, dummy_island_names_2];
    let pickup_dist = 100;
    let islands = [];
    let mounted = false;
    let tweened_bottle_alpha = tweened(1, {duration: 1000, easing: cubicOut});
    $: if (bottle_sprite) bottle_sprite.alpha = $tweened_bottle_alpha;
    $: dist = (mounted) ? Math.sqrt((bottle_sprite.x - ship_sprite.x) ** 2 + (bottle_sprite.y - ship_sprite.y) ** 2) : 1024;
    $: console.log(dist);
    $: if (mounted && dist < pickup_dist && bottle_sprite.visible) {
        console.log("Picked up bottle");
        hide_bottle(); // can't do it inline to avoid circular dependency warning
        // bottle_sprite.visible = false;
        // score += 1;
    }
    function hide_bottle() {
        // bottle_sprite.visible = false;
        tweened_bottle_alpha.set(0.2);
        score += 1;
    }
    function check_legal_move() {
        // don't let the ship go onto land
        let current_pos = {x: ship_sprite.x, y: ship_sprite.y};
        let new_pos = {x: current_pos.x + Math.cos(ship_sprite.rotation + Math.PI / 2) * 10, y: current_pos.y + Math.sin(ship_sprite.rotation + Math.PI / 2) * 10};
        // check if the new position is on any island
        for (let island of islands) {
            // check if the new position is in the island bounding box
            if (island.getBounds().containsPoint(new_pos.x, new_pos.y)) {
                // check each individual land tile
                let land_children_container = island.children[0];
                for (let child of land_children_container.children) {
                    if (child.getBounds().containsPoint(new_pos.x, new_pos.y)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
    onMount(async () => {
        let screen_data = await fetch("/screen1.json");
        screen_data = await screen_data.json();
        console.log(screen_data);

        words_to_vecs = await fetch("/s1.json");
        words_to_vecs = await words_to_vecs.json();
        console.log("revolution", words_to_vecs["revolution"]);

        const app = new PIXI.Application();
        await app.init({ canvas: canvas, width: 1024, height: 1024, backgroundColor: 0x1099bb });

        await PIXI.Assets.init({ manifest: "/manifest.json"}); // need to load manifest before we can reference bundles
        let tiles = await PIXI.Assets.loadBundle("tiles");
        console.log(tiles);

        const tilesheet  = await PIXI.Assets.load('/kenney_pirate/Tilesheet/tiles_sheet.png');
        let frame = tilesheet.frame;
        frame.width = 64;
        frame.height = 64;
        frame.x = 64 * 8;
        frame.y = 64 * 4;
        tilesheet.updateUvs();
        let tiling_sprite = new PIXI.TilingSprite({ texture: tilesheet, width: 2048, height: 2048 });
        app.stage.addChild(tiling_sprite);

        // Create a new Graphics object
        let graphics = new PIXI.Graphics();
        // graphics.moveTo(-200, 600);
        graphics.moveTo(...screen_data.track_bezier[0].start);
        // graphics.bezierCurveTo(100, 100, 200, 200, 1224, 600);
        for (let curve of screen_data.track_bezier) {
            graphics.bezierCurveTo(...curve.cp1, ...curve.cp2, ...curve.end);
        }
        graphics.stroke({width: 140, color: 0xaaaaff, alpha: 0.4});
        app.stage.addChild(graphics);
        //
        for (let island_data of screen_data.islands) {
            console.log("Creating island", island_data);
            let island_container = new PIXI.Container();
            let i1land = new PIXI.Container();
            let i1water = new PIXI.Container();
            island_container.addChild(i1land);
            island_container.addChild(i1water);
            for (let i = 0; i < island_data.tiles.length; i++) {
                for (let j = 0; j < island_data.tiles[i].length; j++) {
                    let island_sprite = PIXI.Sprite.from(tiles[island_data.tiles[i][j]]);
                    island_sprite.x = j * 128;
                    island_sprite.y = i * 128;
                    if (island_data.tiles[i][j] === "water") {
                        // i1water.addChild(island_sprite);
                    } else {
                        i1land.addChild(island_sprite);
                    }
                }
            }
            // add text to the container
            let text = new PIXI.Text("Island", {fontFamily : 'serif', fontSize: 100, fill : 0xff1010, align : 'center'});
            text.x = island_container.width / 2;
            text.y = island_container.height / 2;
            text.anchor.set(0.5);
            island_container.addChild(text);
            // pos
            island_container.position = island_data.position;
            island_container.pivot.set(island_container.width / 2, island_container.height / 2);
            island_container.rotation = island_data.rotation;
            // island_container.rotation = 3.14 / 6.0;
            // scale down 50%
            island_container.scale.set(0.5);
            islands.push(island_container);
            app.stage.addChild(island_container);
        }

        // create bottle sprite
        bottle_sprite = PIXI.Sprite.from(tiles.bottle);
        bottle_sprite.position = screen_data.bottle;
        bottle_sprite.anchor.set(0.5);
        app.stage.addChild(bottle_sprite);
        

        // ship sprite
        await PIXI.Assets.load('/kenney_pirate/PNG/Retina/Ships/ship (1).png');
        ship_sprite = PIXI.Sprite.from('/kenney_pirate/PNG/Retina/Ships/ship (1).png');
        ship_sprite.anchor.set(0.5);
        ship_sprite.x = 1024 / 2;
        ship_sprite.y = 1024 / 2;
        app.stage.addChild(ship_sprite);

        // app.stage.addChild(sprite);
        app.ticker.add((ticker) => {
            // tilesheet_sprite.rotation +=  0.01 * ticker.deltaTime;
        });

        window.addEventListener('keydown', handleKeyPress);
        mounted = true;
        return () => {
            app.destroy();
            window.removeEventListener('keydown', handleKeyPress); 
        };
    });
    function handleKeyPress(event) {
        // right
        let key_to_rotation = {
            ArrowDown: 0,
            ArrowUp: Math.PI,
            ArrowLeft: Math.PI / 2,
            ArrowRight: -Math.PI / 2
        };
        if (event.key in key_to_rotation) {
            ship_sprite.rotation = key_to_rotation[event.key];
            if (check_legal_move()) {
                ship_sprite.x += Math.cos(ship_sprite.rotation + Math.PI / 2) * 10;
                ship_sprite.y += Math.sin(ship_sprite.rotation + Math.PI / 2) * 10;
            }
            event.preventDefault();
        }
    }
    function handleEnter(event) {
        if (event.key === "Enter") {
            console.log("Enter pressed", event.target.value, words_to_vecs[event.target.value]);
            let delta = words_to_vecs[event.target.value];
            let factor = 20.0;
            // ship_sprite.x += delta[0] * factor;
            tweened_ship_pos.set({x: ship_sprite.x + delta[0] * factor, y: ship_sprite.y + delta[1] * factor});
            // ship_sprite.y += delta[1] * factor;
            event.preventDefault();
        }
    }
    function handleReceiveMessage(event) {
        console.log("Received message", event.detail.message);
        let delta = words_to_vecs[event.detail.message];
        let factor = 20.0;
        tweened_ship_pos.set({x: ship_sprite.x + delta[0] * factor, y: ship_sprite.y + delta[1] * factor});
        // ship_sprite.x += delta[0] * factor;
        // ship_sprite.y += delta[1] * factor;
    }

</script>

<div class="hidden">
    <Comm messages={["m1", "m2", "m3"]} on:receive_message={handleReceiveMessage} />
</div>

<div class="game h-screen w-screen bg-gray-500 flex flex-row items-center justify-center relative" >
    {#if bottle_sprite}
        <input type="range" bind:value={bottle_sprite.x} min="0" max="1024" step="1" class="absolute"/>
    {/if}
    <div class="h-[1024px] flex flex-row gap-8">
        <canvas class="h-full w-full border-4 border-black" id="app" bind:this={canvas}></canvas>
        <div class="h-full w-[400px] bg-green-300 border-4 border-black text-3xl">
            <input type="text" class="w-full" placeholder="Type here" on:keydown={handleEnter}
            />
        </div>
    </div>
</div>
